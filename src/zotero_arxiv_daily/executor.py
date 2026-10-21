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


# 行人重识别、行人检索相关词（匹配时忽略大小写）
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
    "re-id",
    "Reid",
    "ReID",
    "Re-ID",
    "Re-id",
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


def is_zotero_unavailable_error(exc: Exception) -> bool:
    """Recognize HTTP 503 and Zotero's explicit maintenance response, even on 400."""
    if is_zotero_503_error(exc):
        return True

    current = exc
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        message = str(current).casefold().replace("’", "'")
        if "zotero's servers are temporarily offline" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def is_zotero_503_error(exc: Exception) -> bool:
    """检查异常及其异常链是否表示 HTTP 503。"""
    current = exc
    visited = set()

    while current is not None and id(current) not in visited:
        visited.add(id(current))

        # 检查原始 HTTP 异常携带的响应状态
        response = getattr(current, "response", None)
        status_code = getattr(response, "status_code", None)

        if status_code == 503:
            return True

        # 部分异常直接携带状态码
        if getattr(current, "status_code", None) == 503:
            return True

        # pyzotero 包装后的异常可能只保留错误文字。
        # 只匹配明确的 HTTP 状态描述，避免误匹配其他数字。
        message = str(current)

        if re.search(
            r"\b(?:code|status(?:[_\s-]*code)?)"
            r"\s*[:=]?\s*['\"]?503\b"
            r"|\bserver error\s*['\"]?503\b"
            r"|\b503\s+service unavailable\b",
            message,
            flags=re.IGNORECASE,
        ):
            return True

        current = current.__cause__ or current.__context__

    return False


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

        # 延迟初始化：只有 Zotero 可用且有匹配论文时才加载排序器
        self.reranker = None

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
        corpus = []
        zotero_unavailable = False

        try:
            corpus = self.fetch_zotero_corpus()

        except Exception as exc:
            # Zotero 的明确维护响应可能使用 HTTP 400，而不是 503。
            if not is_zotero_unavailable_error(exc):
                raise

            zotero_unavailable = True

            logger.warning(
                "Zotero is temporarily unavailable "
                "(HTTP 503 or an explicit maintenance response). "
                "Continuing without Zotero for this run."
            )

        if not zotero_unavailable:
            corpus = self.filter_corpus(corpus)

            if not corpus:
                logger.error(
                    "No Zotero papers found. Please check your "
                    "Zotero library, abstracts and collection filters."
                )
                return

        # 2. 获取各来源的新论文
        all_papers = []

        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()

            if not papers:
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

        # 3. 根据标题和摘要筛选多模态 ReID 论文
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

        selected_papers = []

        if all_papers:
            # 4. 正常时按 Zotero 相似度排序；
            #    Zotero 维护时保留检索返回顺序
            if zotero_unavailable:
                logger.info(
                    "Skipping Zotero similarity reranking. "
                    "Using retrieval order."
                )
                selected_papers = all_papers

            else:
                if self.reranker is None:
                    self.reranker = get_reranker_cls(
                        self.config.executor.reranker
                    )(self.config)

                logger.info(
                    "Reranking papers using Zotero corpus..."
                )

                selected_papers = self.reranker.rerank(
                    all_papers,
                    corpus,
                )

            selected_papers = selected_papers[
                :self.config.executor.max_paper_num
            ]

            # 5. 生成摘要和作者单位
            logger.info("Generating TLDR and affiliations...")

            for paper in tqdm(selected_papers):
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
        email_content = render_email(selected_papers)
        send_email(self.config, email_content)
        logger.info("Email sent successfully")
