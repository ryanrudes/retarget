"""Typed registries for extension points."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from enum import StrEnum
from typing import Generic, TypeVar

T = TypeVar("T")
RegistryKey = str | StrEnum


class Registry(Generic[T]):
    """Small decorator-friendly registry."""

    def __init__(self, name: str, *, decorator_transform: Callable[[object], T] | None = None) -> None:
        self.name = name
        self._decorator_transform = decorator_transform
        self._items: dict[str, T] = {}

    def register(self, key: RegistryKey, value: T | None = None, *, replace: bool = False) -> Callable[[T], T] | T:
        """Register a value directly or as a decorator."""

        normalized = _normalize_key(key)
        if not normalized:
            raise ValueError(f"{self.name} registry keys must not be empty")

        def decorator(item: T) -> T:
            if not replace and normalized in self._items:
                raise KeyError(f"{normalized!r} is already registered in {self.name}")
            self._items[normalized] = (
                self._decorator_transform(item) if self._decorator_transform is not None else item
            )
            return item

        if value is None:
            return decorator
        return decorator(value)

    def get(self, key: RegistryKey) -> T:
        """Return a registered value."""

        normalized = _normalize_key(key)
        try:
            return self._items[normalized]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<none>"
            raise KeyError(f"Unknown {self.name} key {normalized!r}. Available: {available}") from exc

    def maybe_get(self, key: RegistryKey) -> T | None:
        """Return a registered value or `None`."""

        return self._items.get(_normalize_key(key))

    def names(self) -> tuple[str, ...]:
        """Registered names in sorted order."""

        return tuple(sorted(self._items))

    def values(self) -> tuple[T, ...]:
        """Registered values ordered by sorted key."""

        return tuple(self._items[name] for name in self.names())

    def items(self) -> tuple[tuple[str, T], ...]:
        """Registered `(name, value)` pairs ordered by name."""

        return tuple((name, self._items[name]) for name in self.names())

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str | StrEnum):
            return False
        return _normalize_key(key) in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._items)

    def require_all(self, keys: Iterable[RegistryKey]) -> None:
        """Validate that every key exists."""

        missing = self.missing(keys)
        if missing:
            available = ", ".join(self.names()) or "<none>"
            raise KeyError(f"Missing {self.name} registrations: {', '.join(missing)}. Available: {available}")

    def missing(self, keys: Iterable[RegistryKey]) -> tuple[str, ...]:
        """Return missing keys in first-seen order."""

        missing: list[str] = []
        seen: set[str] = set()
        for key in keys:
            normalized = _normalize_key(key)
            if normalized in self._items or normalized in seen:
                continue
            missing.append(normalized)
            seen.add(normalized)
        return tuple(missing)


def _normalize_key(key: RegistryKey) -> str:
    return key.value.strip() if isinstance(key, StrEnum) else key.strip()
