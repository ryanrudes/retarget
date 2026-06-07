"""Typed registries for extension points."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from enum import StrEnum
from typing import Generic, TypeVar

T = TypeVar("T")
K = TypeVar("K", bound=StrEnum)


class Registry(Generic[K, T]):
    """Small decorator-friendly registry.

    Attributes:
        name (str): Human-readable registry label used in error messages.
    """

    def __init__(
        self,
        name: str,
        key_type: type[K],
        *,
        decorator_transform: Callable[[object], T] | None = None,
    ) -> None:
        """Create an empty registry.

        Args:
            name (str): Registry label included in lookup errors.
            decorator_transform (Callable[[object], T] | None): Optional wrapper applied
                when registering via the decorator form.
        """
        self.name = name
        self.key_type = key_type
        self._decorator_transform = decorator_transform
        self._items: dict[tuple[type[StrEnum], str], tuple[K, T]] = {}
        self._serialized: dict[str, tuple[type[StrEnum], str]] = {}

    def register(self, key: K, value: T | None = None, *, replace: bool = False) -> Callable[[T], T] | T:
        """Register a value directly or as a decorator.

        Args:
            key (K): Typed enum member used as the registry key.
            value (T | None): Object to register; omit to use as ``@registry.register(...)``.
            replace (bool): Allow overwriting an existing key.

        Returns:
            Callable[[T], T] | T: The decorator when ``value`` is omitted, otherwise the
                registered value.
        """

        if not isinstance(key, self.key_type):
            raise TypeError(f"{self.name} registry keys must subclass {self.key_type.__name__}")
        normalized = key.value.strip()
        if not normalized:
            raise ValueError(f"{self.name} registry keys must not be empty")
        identity = _identity(key)

        def decorator(item: T) -> T:
            existing_serialized = self._serialized.get(normalized)
            if existing_serialized is not None and existing_serialized != identity:
                raise KeyError(
                    f"{normalized!r} is already used by a different enum vocabulary in {self.name}"
                )
            if not replace and identity in self._items:
                raise KeyError(f"{type(key).__name__}.{key.name} is already registered in {self.name}")
            registered = (
                self._decorator_transform(item) if self._decorator_transform is not None else item
            )
            self._items[identity] = (key, registered)
            self._serialized[normalized] = identity
            return item

        if value is None:
            return decorator
        return decorator(value)

    def get(self, key: K) -> T:
        """Return a registered value.

        Args:
            key (K): Typed enum member to look up.

        Returns:
            T: Registered value for ``key``.

        Raises:
            KeyError: If ``key`` is not registered.
        """

        if not isinstance(key, self.key_type):
            raise TypeError(f"{self.name} keys must be {self.key_type.__name__} members")
        try:
            return self._items[_identity(key)][1]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<none>"
            raise KeyError(
                f"Unknown {self.name} key {type(key).__name__}.{key.name}. Available: {available}"
            ) from exc

    def get_serialized(self, value: str) -> T:
        """Resolve a serialized key at a configuration or CLI boundary."""

        normalized = value.strip()
        try:
            return self._items[self._serialized[normalized]][1]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<none>"
            raise KeyError(f"Unknown {self.name} key {normalized!r}. Available: {available}") from exc

    def key_from_serialized(self, value: str) -> K:
        """Return the registered enum member for a serialized value."""

        normalized = value.strip()
        try:
            return self._items[self._serialized[normalized]][0]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<none>"
            raise KeyError(f"Unknown {self.name} key {normalized!r}. Available: {available}") from exc

    def maybe_get(self, key: K) -> T | None:
        """Return a registered value or ``None``.

        Args:
            key (K): Typed enum member to look up.

        Returns:
            T | None: Registered value, or ``None`` when ``key`` is absent.
        """

        if not isinstance(key, self.key_type):
            raise TypeError(f"{self.name} keys must be {self.key_type.__name__} members")
        item = self._items.get(_identity(key))
        return None if item is None else item[1]

    def names(self) -> tuple[str, ...]:
        """Registered names in sorted order."""

        return tuple(sorted(self._serialized))

    def values(self) -> tuple[T, ...]:
        """Registered values ordered by sorted key."""

        return tuple(self._items[self._serialized[name]][1] for name in self.names())

    def items(self) -> tuple[tuple[str, T], ...]:
        """Registered `(name, value)` pairs ordered by name."""

        return tuple((name, self._items[self._serialized[name]][1]) for name in self.names())

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, self.key_type):
            return False
        return _identity(key) in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._items)

    def require_all(self, keys: Iterable[K]) -> None:
        """Validate that every key exists.

        Args:
            keys (Iterable[K]): Typed keys that must be registered.

        Raises:
            KeyError: If any key in ``keys`` is missing.
        """

        missing = self.missing(keys)
        if missing:
            available = ", ".join(self.names()) or "<none>"
            raise KeyError(f"Missing {self.name} registrations: {', '.join(missing)}. Available: {available}")

    def missing(self, keys: Iterable[K]) -> tuple[str, ...]:
        """Return missing keys in first-seen order.

        Args:
            keys (Iterable[K]): Typed keys to check for registration.

        Returns:
            tuple[str, ...]: Normalized keys from ``keys`` that are not registered,
                preserving first-seen order without duplicates.
        """

        missing: list[str] = []
        seen: set[str] = set()
        for key in keys:
            if not isinstance(key, self.key_type):
                raise TypeError(f"{self.name} keys must be {self.key_type.__name__} members")
            normalized = key.value.strip()
            if _identity(key) in self._items or normalized in seen:
                continue
            missing.append(normalized)
            seen.add(normalized)
        return tuple(missing)


def _identity(key: StrEnum) -> tuple[type[StrEnum], str]:
    return type(key), key.value.strip()
