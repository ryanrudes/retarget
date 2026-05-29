"""Typed registries for extension points."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Small decorator-friendly registry."""

    def __init__(self, name: str, *, decorator_transform: Callable[[object], T] | None = None) -> None:
        self.name = name
        self._decorator_transform = decorator_transform
        self._items: dict[str, T] = {}

    def register(self, key: str, value: T | None = None, *, replace: bool = False) -> Callable[[T], T] | T:
        """Register a value directly or as a decorator."""

        normalized = key.strip()
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

    def get(self, key: str) -> T:
        """Return a registered value."""

        try:
            return self._items[key]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<none>"
            raise KeyError(f"Unknown {self.name} key {key!r}. Available: {available}") from exc

    def maybe_get(self, key: str) -> T | None:
        """Return a registered value or `None`."""

        return self._items.get(key)

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
        return key in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._items)

    def require_all(self, keys: Iterable[str]) -> None:
        """Validate that every key exists."""

        missing = self.missing(keys)
        if missing:
            available = ", ".join(self.names()) or "<none>"
            raise KeyError(f"Missing {self.name} registrations: {', '.join(missing)}. Available: {available}")

    def missing(self, keys: Iterable[str]) -> tuple[str, ...]:
        """Return missing keys in first-seen order."""

        missing: list[str] = []
        seen: set[str] = set()
        for key in keys:
            if key in self._items or key in seen:
                continue
            missing.append(key)
            seen.add(key)
        return tuple(missing)
