# Export types

Typed export configuration and results. Built-in exporters register under `ExportFormat` keys; see [Registries](registries.md) for `retarget.export.exporters`.

## Spec & result

::: retarget.export.spec.ExportSpec

::: retarget.export.spec.ExportResult

## Exporter protocol

Implementations register on [`retarget.export.exporters`][retarget.export.exporters]. The interface is [`Exporter`][retarget.core.protocols.Exporter] (documented on [Protocols](protocols.md)).
