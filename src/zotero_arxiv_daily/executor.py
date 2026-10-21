import random
import re
from datetime import datetime

from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from openai import OpenAI
from tqdm import tqdm

from .utils import glob_match, send_email
from .retriever import get_retriever_cls
from .protocol import CorpusPaper
from .reranker import get_reranker_cls
from .construct_email import render_email


# 行人重识别、行人检索相关词
REID_KEYWORDS = (
    "person re-identification",
    "person reidentification",
    "person re identification",
    "person re-id",
    "pedestrian re-identification",
    "pedestrian reidentification",
    "pedestrian re identification",
    "person retrieval",
    "pedestrian retrieval",
    "reid",
    "Reid",
    "ReID",
    "re-id",
    "行人重识别",
    "行人再识别",
    "行人检索",
)

# 多模态、跨模态及具体模态组合相关词
MODALITY_KEYWORDS = (
    "multimodal",
    "multi-modal",
    "multi modal",
    "cross-modal",
    "cross modal",
    "cross-modality",
    "cross modality",
    "visible-infrared",
    "visible infrared",
    "infrared-visible",
    "infrared visible",
    "rgb-infrared",
    "rgb infrared",
    "rgb-ir",
    "rgb ir",
    "text-to-image",
    "text to image",
    "text-based",
    "text based",
    "text-guided",
    "text guided",
    "image-text",
    "image text",
    "multispectral",
    "multi-spectral",
    "multi spectral",
    "多模态",
    "跨模态",
    "可见光红外",
    "可见光与红外",
    "文本行人",
)


def normalize_keyword_text(text: str) -> str:
    """统一大小写、连接号和空白。"""
    text = (text or "").casefold()
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_keyword(text: str, keyword: str) -> bool:
    """英文关键词按完整词匹配，中文关键词按包含关系匹配。"""
    keyword = normalize_keyword_text(keyword)

    if keyword.isascii():
        pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
        return re.search(pattern, text) is not None

    return keyword in text


def is_multimodal_reid_paper(paper) -> bool:
    """标题和摘要合并后，必须同时命中两组关键词。"""
    text = normalize_keyword_text(
        f"{paper.title or ''}\n{paper.abstract or ''}"
    )

    has_reid = any(
        contains_keyword(text, keyword)
        for keyword in REID_KEYWORDS
    )
    has_modality = any(
        contains_keyword(text, keyword)
        for keyword in MODALITY_KEYWORDS
    )

    return has_reid and has_modality


def normalize_path_patterns(
    patterns: list[str] | ListConfig | None,
    config_key: str,
) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns "
            'or null, for example ["2026/survey/**"]. '
            "Single strings are not supported."
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(
            f"config.zotero.{config_key} must contain only "
            "glob pattern strings."
        )

    return list(patterns)


class Executor:
    def __init__(self, config: DictConfig):
        self.config = config

        self.include_path_patterns = normalize_path_patterns(
            config.zotero.include_path,
            "include_path",
        )
        self.ignore_path_patterns = normalize_path_patterns(
            config.zotero.ignore_path,
            "ignore_path",
        )

        self.retrievers = {
            source: get_retriever_cls(source)(config)
            for source in config.executor.source
        }

        self.reranker = get_reranker_cls(
            config.executor.reranker
        )(config)

        self.openai_client = OpenAI(
            api_key=config.llm.api.key,
            base_url=config.llm.api.base_url,
        )

    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")

        zot = zotero.Zotero(
            self.config.zotero.user_id,
            "user",
            self.config.zotero.api_key,
        )

        collections = zot.everything(zot.collections())
        collections = {
            collection["key"]: collection
            for collection in collections
        }

        corpus = zot.everything(
            zot.items(
                itemType="conferencePaper || journalArticle || preprint"
            )
        )

        corpus = [
            item for item in corpus
            if item["data"]["abstractNote"] != ""
        ]

        def get_collection_path(col_key: str) -> str:
            collection = collections[col_key]["data"]
            parent = collection["parentCollection"]

            if parent:
                return (
                    get_collection_path(parent)
                    + "/"
                    + collection["name"]
                )

            return collection["name"]

        for item in corpus:
            item["paths"] = [
                get_collection_path(col)
                for col in item["data"]["collections"]
            ]

        logger.info(f"Fetched {len(corpus)} zotero papers")

        return [
            CorpusPaper(
                title=item["data"]["title"],
                abstract=item["data"]["abstractNote"],
                added_date=datetime.strptime(
                    item["data"]["dateAdded"],
                    "%Y-%m-%dT%H:%M:%SZ",
                ),
                paths=item["paths"],
            )
            for item in corpus
        ]

    def filter_corpus(
        self,
        corpus: list[CorpusPaper],
    ) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(
                "Selecting zotero papers matching include_path: "
                f"{self.include_path_patterns}"
            )

            corpus = [
                paper for paper in corpus
                if any(
                    glob_match(path, pattern)
                    for path in paper.paths
                    for pattern in self.include_path_patterns
                )
            ]

        if self.ignore_path_patterns:
            logger.info(
                "Excluding zotero papers matching ignore_path: "
                f"{self.ignore_path_patterns}"
            )

            corpus = [
                paper for paper in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in paper.paths
                    for pattern in self.ignore_path_patterns
                )
            ]

        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(
                corpus,
                min(5, len(corpus)),
            )
            sample_text = "\n".join(
                paper.title + " - " + "\n".join(paper.paths)
                for paper in samples
            )

            logger.info(
                f"Selected {len(corpus)} zotero papers:\n"
                f"{sample_text}\n..."
            )

        return corpus

    def run(self):
        # 1. 获取 Zotero 参考论文
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)

        if len(corpus) == 0:
            logger.error(
                "No zotero papers found. Please check your "
                "Zotero library, abstracts and collection filters."
            )
            return

        # 2. 获取各来源的新论文
        all_papers = []

        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()

            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue

            logger.info(
                f"Retrieved {len(papers)} {source} papers"
            )
            all_papers.extend(papers)

        logger.info(
            f"Total {len(all_papers)} papers retrieved "
            "from all sources"
        )

        # 3. 根据标题和摘要，筛选多模态 ReID 论文
        before_filter = len(all_papers)

        all_papers = [
            paper for paper in all_papers
            if is_multimodal_reid_paper(paper)
        ]

        logger.info(
            f"Multimodal ReID filter: "
            f"{before_filter} -> {len(all_papers)} papers"
        )

        for paper in all_papers:
            logger.info(f"Matched paper: {paper.title}")

        # 4. 用 Zotero 参考论文对筛选结果排序
        reranked_papers = []

        if len(all_papers) > 0:
            logger.info("Reranking papers...")

            reranked_papers = self.reranker.rerank(
                all_papers,
                corpus,
            )
            reranked_papers = reranked_papers[
                :self.config.executor.max_paper_num
            ]

            # 5. 生成摘要和作者单位
            logger.info("Generating TLDR and affiliations...")

            for paper in tqdm(reranked_papers):
                paper.generate_tldr(
                    self.openai_client,
                    self.config.llm,
                )
                paper.generate_affiliations(
                    self.openai_client,
                    self.config.llm,
                )

        elif not self.config.executor.send_empty:
            logger.info(
                "No papers matched the multimodal ReID filter. "
                "No email will be sent."
            )
            return

        # 6. 发送邮件
        logger.info("Sending email...")
        email_content = render_email(reranked_papers)
        send_email(self.config, email_content)
        logger.info("Email sent successfully")
