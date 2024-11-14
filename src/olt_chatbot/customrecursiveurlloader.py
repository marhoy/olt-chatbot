"""New class for overriding method in RecursiveUrlLoader.

It is used to recursively crawl URLs and extract documents from the web pages.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import requests
from langchain_community.document_loaders.recursive_url_loader import (
    RecursiveUrlLoader,
)
from langchain_core.documents import Document
from langchain_core.utils.html import extract_sub_links
from loguru import logger


class CustomRecursiveUrlLoader(RecursiveUrlLoader):
    """Custom RecursiveUrlLoader class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the CustomRecursiveUrlLoader.

        Args:
            *args: Positional arguments passed to the parent class.
            **kwargs: Keyword arguments passed to the parent class.
        """
        super().__init__(*args, **kwargs)

    def _get_child_links_recursive(
        self, url: str, visited: set[str], *, depth: int = 0
    ) -> Iterator[Document]:
        """Recursively get all child links starting with the path of the input URL.

        Args:
            url: The URL to crawl.
            visited: A set of visited URLs.
            depth: Current depth of recursion. Stop when depth >= max_depth.
        """
        if depth >= self.max_depth:
            return

        # Get all links that can be accessed from the current URL
        visited.add(url)
        try:
            response = requests.get(url, timeout=self.timeout, headers=self.headers)

            if self.encoding is not None:
                response.encoding = self.encoding
            elif self.autoset_encoding:
                response.encoding = response.apparent_encoding

            if self.check_response_status and 400 <= response.status_code <= 599:
                raise ValueError(f"Received HTTP status {response.status_code}")
        except Exception as e:
            if self.continue_on_failure:
                logger.warning(
                    f"Unable to load from {url}. Received error {e} of type "
                    f"{e.__class__.__name__}"
                )
                return
            else:
                raise e
        content = self.extractor(response)
        if content:
            yield Document(
                page_content=content,
                metadata=self.metadata_extractor(response.text, url, response),
            )

        # Store the visited links and recursively visit the children
        sub_links = extract_sub_links(
            response.text,
            url,
            base_url=self.base_url,
            pattern=self.link_regex,
            prevent_outside=self.prevent_outside,
            exclude_prefixes=self.exclude_dirs,
            continue_on_failure=self.continue_on_failure,
        )
        for link in sub_links:
            # Check all unvisited links
            if link not in visited:
                yield from self._get_child_links_recursive(
                    link, visited, depth=depth + 1
                )
