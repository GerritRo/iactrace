from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..core.obstructions import ObstructionGroup
    from ..sensors import SensorGroup
    from ..telescope.lenses import LensGroup
    from ..telescope.mirrors import MirrorGroup

T = TypeVar("T")


@runtime_checkable
class ConfigSerializable(Protocol):
    """Protocol for classes that can be serialized to/from config dicts.

    Classes implementing this protocol can be automatically handled by
    the YAML loader and dumper without modification to those modules.
    """

    #: The type identifier used in YAML configs (e.g., "circular", "cylinder")
    config_type: ClassVar[str]

    def to_config(self, index: int) -> dict[str, Any]:
        """Convert a single element at index to a config dict."""
        ...

    @classmethod
    def from_config(cls, configs: list[dict[str, Any]], **kwargs: Any) -> ConfigSerializable:
        """Create an instance from a list of config dicts."""
        ...


class TypeRegistry(Generic[T]):
    """Registry mapping type names to classes for serialization."""

    def __init__(self, name: str) -> None:
        """Initialize a new registry.

        Args:
            name: Human-readable name for this registry (for error messages).
        """
        self.name = name
        self._types: dict[str, type[T]] = {}

    def register(self, type_name: str) -> Callable[[type[T]], type[T]]:
        """Decorator to register a class with a type name.

        Args:
            type_name: The type identifier used in YAML configs.

        Returns:
            Decorator that registers the class and sets its config_type.

        Example:
            @obstruction_registry.register("cylinder")
            class CylinderGroup(ObstructionGroup):
                ...
        """

        def decorator(cls: type[T]) -> type[T]:
            if type_name in self._types:
                raise ValueError(
                    f"Type '{type_name}' already registered in {self.name} "
                    f"registry as {self._types[type_name].__name__}"
                )
            self._types[type_name] = cls
            cls.config_type = type_name  # type: ignore[attr-defined]
            return cls

        return decorator

    def get(self, type_name: str) -> type[T]:
        """Get the class for a type name.

        Args:
            type_name: The type identifier from YAML config.

        Returns:
            The registered class.

        Raises:
            KeyError: If type_name is not registered.
        """
        # Ensure registries are populated on first access
        _ensure_registries_populated()

        if type_name not in self._types:
            available = ", ".join(sorted(self._types.keys()))
            raise KeyError(
                f"Unknown {self.name} type '{type_name}'. "
                f"Available types: {available}"
            )
        return self._types[type_name]

    def get_type_name(self, instance: Any) -> str:
        """Get the type name for an instance."""
        if hasattr(instance, "config_type"):
            return instance.config_type

        for type_name, cls in self._types.items():
            if isinstance(instance, cls):
                return type_name

        raise ValueError(
            f"Instance of {type(instance).__name__} is not registered "
            f"in {self.name} registry"
        )

    def is_registered(self, type_name: str) -> bool:
        """Check if a type name is registered."""
        return type_name in self._types

    def registered_types(self) -> list[str]:
        """Return list of all registered type names."""
        _ensure_registries_populated()
        return list(self._types.keys())

    def __contains__(self, type_name: str) -> bool:
        return type_name in self._types

    def __repr__(self) -> str:
        types = ", ".join(sorted(self._types.keys()))
        return f"TypeRegistry({self.name!r}, types=[{types}])"


# Global registries for each component category
mirror_registry: TypeRegistry[MirrorGroup] = TypeRegistry("mirror")
lens_registry: TypeRegistry[LensGroup] = TypeRegistry("lens")
obstruction_registry: TypeRegistry[ObstructionGroup] = TypeRegistry("obstruction")
sensor_registry: TypeRegistry[SensorGroup] = TypeRegistry("sensor")


def get_all_registries() -> dict[str, TypeRegistry[Any]]:
    """Return all component registries."""
    return {
        "mirror": mirror_registry,
        "lens": lens_registry,
        "obstruction": obstruction_registry,
        "sensor": sensor_registry,
    }


_registries_populated = False


def _ensure_registries_populated() -> None:
    """Ensure all component types are registered.

    This function is idempotent - it only populates the registries once.
    Called automatically when accessing registries.
    """
    global _registries_populated
    if _registries_populated:
        return

    # Import and register obstruction types
    from ..core.obstructions import (
        BoxGroup,
        CylinderGroup,
        OpenCylinderGroup,
        OrientedBoxGroup,
        SphereGroup,
        TriangleGroup,
    )

    for cls in [CylinderGroup, OpenCylinderGroup, BoxGroup, SphereGroup, OrientedBoxGroup, TriangleGroup]:
        config_type: str = cls.config_type  # type: ignore[attr-defined]
        if config_type not in obstruction_registry:
            obstruction_registry._types[config_type] = cls  # type: ignore[assignment]

    # Import and register sensor types
    from ..sensors.hexagonal import HexagonalSensorGroup
    from ..sensors.square import SquareSensorGroup

    for cls in [SquareSensorGroup, HexagonalSensorGroup]:
        config_type = cls.config_type  # type: ignore[attr-defined]
        if config_type not in sensor_registry:
            sensor_registry._types[config_type] = cls  # type: ignore[assignment]

    # Import and register mirror types
    from ..telescope.mirrors import AsphericDiskMirrorGroup, AsphericPolygonMirrorGroup

    for cls in [AsphericDiskMirrorGroup, AsphericPolygonMirrorGroup]:
        config_type = cls.config_type  # type: ignore[attr-defined]
        if config_type not in mirror_registry:
            mirror_registry._types[config_type] = cls  # type: ignore[assignment]

    # Import and register lens types
    from ..telescope.lenses import AsphericDiskLensGroup, PlanoSlabGroup

    for cls in [AsphericDiskLensGroup, PlanoSlabGroup]:
        config_type = cls.config_type  # type: ignore[attr-defined]
        if config_type not in lens_registry:
            lens_registry._types[config_type] = cls  # type: ignore[assignment]

    _registries_populated = True
