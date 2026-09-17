from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from face_preprocess.config import QCConfig, QCRule
from face_preprocess.qc.registry import empirical_group_for_metric
from face_preprocess.types import Measurement


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    metric: str
    evaluated: bool
    passed: bool | None
    triggered: bool
    severity: str
    reason: str
    observed: Any = None


@dataclass(frozen=True)
class QCAssessment:
    status: str
    qc_incomplete: bool
    results: tuple[RuleResult, ...]
    hard_failures: tuple[str, ...]


def _matches(value: Any, operator: str, threshold: Any) -> bool:
    if operator == "equals":
        return value == threshold
    if operator == "gt":
        return float(value) > float(threshold)
    if operator == "gte":
        return float(value) >= float(threshold)
    if operator == "lt":
        return float(value) < float(threshold)
    if operator == "lte":
        return float(value) <= float(threshold)
    if operator in {"inside", "outside"}:
        if not isinstance(threshold, (list, tuple)) or len(threshold) != 2:
            raise ValueError(f"{operator} threshold must be [lower, upper]")
        inside = float(threshold[0]) <= float(value) <= float(threshold[1])
        return inside if operator == "inside" else not inside
    raise ValueError(f"unsupported QC operator: {operator}")


def _measurement(value: object) -> Measurement:
    if isinstance(value, Measurement):
        return value
    if isinstance(value, Mapping):
        return Measurement(
            value=value.get("value"),
            unit=str(value.get("unit", "")),
            compute_status=str(value.get("compute_status", "computed")),
            error=str(value["error"]) if value.get("error") is not None else None,
        )
    raise TypeError("QC measurement must be Measurement or a mapping")


def _missing_result(rule: QCRule, reason: str) -> RuleResult:
    return RuleResult(
        rule_id=rule.rule_id,
        metric=rule.metric,
        evaluated=False,
        passed=None,
        triggered=True,
        severity="warning",
        reason=reason,
    )


def evaluate_qc(
    measurements: Mapping[str, object],
    config: QCConfig,
    *,
    level: str,
    hard_failures: tuple[str, ...] | list[str] = (),
) -> QCAssessment:
    if level not in {"standard", "enhanced"}:
        raise ValueError(f"invalid QC level: {level}")
    results: list[RuleResult] = []
    incomplete = False
    for raw in measurements.values():
        try:
            item = _measurement(raw)
        except (TypeError, ValueError):
            incomplete = True
            continue
        if item.compute_status in {"error", "computation_error", "not_computed"}:
            incomplete = True
    for rule in config.rules:
        if rule.minimum_qc_level == "enhanced" and level != "enhanced":
            continue
        group = empirical_group_for_metric(rule.metric)
        mode = config.metric_groups.get(group or "", "record_only")
        if mode == "disabled":
            continue
        if rule.metric not in measurements:
            incomplete = True
            results.append(_missing_result(rule, "measurement missing"))
            continue
        try:
            measurement = _measurement(measurements[rule.metric])
        except (TypeError, ValueError) as exc:
            incomplete = True
            results.append(_missing_result(rule, f"invalid measurement: {exc}"))
            continue
        if measurement.compute_status != "computed":
            incomplete = True
            results.append(
                _missing_result(
                    rule,
                    f"measurement {measurement.compute_status}: {measurement.error or 'no detail'}",
                )
            )
            continue
        if measurement.unit != rule.unit:
            incomplete = True
            results.append(
                _missing_result(
                    rule,
                    f"unit mismatch: expected {rule.unit}, got {measurement.unit}",
                )
            )
            continue
        try:
            matched = _matches(measurement.value, rule.operator, rule.threshold)
        except (TypeError, ValueError, OverflowError) as exc:
            incomplete = True
            results.append(_missing_result(rule, f"rule evaluation error: {exc}"))
            continue
        triggered = matched
        passed = not triggered
        severity = "record_only" if mode == "record_only" else rule.severity
        if rule.minimum_qc_level == "enhanced" and severity != "record_only":
            severity = "warning"
        results.append(
            RuleResult(
                rule_id=rule.rule_id,
                metric=rule.metric,
                evaluated=True,
                passed=passed,
                triggered=triggered,
                severity=severity,
                reason="trigger condition matched" if triggered else "trigger condition not matched",
                observed=measurement.value,
            )
        )
    hard = tuple(str(item) for item in hard_failures)
    if hard:
        status = "failed"
    elif incomplete or any(item.triggered and item.severity == "warning" for item in results):
        status = "warning"
    else:
        status = "passed"
    return QCAssessment(
        status=status,
        qc_incomplete=incomplete,
        results=tuple(results),
        hard_failures=hard,
    )
