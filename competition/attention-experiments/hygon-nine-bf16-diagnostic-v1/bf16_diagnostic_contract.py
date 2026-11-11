"""Completion means observations collected, never numerical correctness acceptance."""
import math


def json_preview(values):
    return [v if math.isfinite(v) else str(v) for v in values]


def validate_diagnostic(report):
    if (report.get('result')!='DIAGNOSTIC_COMPLETE' or report.get('correctness_passed') is not False
            or report.get('target',{}).get('arch')!='gfx936'
            or sorted(c.get('name') for c in report.get('conversion',[]))!=['fp32-direct','fp64-direct','fp64-via-fp32']
            or [c.get('round') for c in report.get('attention',[])]!=[1,2]):
        raise ValueError('incomplete diagnostic observations / invalid correctness claim')
    for case in report['conversion']:
        if type(case.get('passed')) is not bool:raise ValueError('missing conversion outcome')
    for case in report['attention']:
        if (any(type(case.get(k)) is not bool for k in ('aot_passed','jit_passed'))
                or case.get('inputs_unchanged') is not True or case.get('guards_unchanged') is not True):
            raise ValueError('missing pair observation or damaged memory')
