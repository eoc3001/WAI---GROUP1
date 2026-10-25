"""Agent layer: how strongly CycleSignal recommends a GP visit, and why.

agent_level mirrors model1/model1_pipeline_v6.py agent_level (that file runs as a
script on import, so it cannot be imported). Keep the two in sync.
The plain-language assistant is in assistant.py; the trusted reading list in resources.py.
"""


def agent_level(model2_flag=False, model2_flag_cycle_neutral=None, scan_pcom=None,
                axis1_pattern=False, axis1_infrequent=False, axis2=False, no_period_90=False,
                years_since_menarche=None, hormonal_contraception=False, pregnant_or_breastfeeding=False,
                axis2_allowed=False):
    """Mirror of model1_pipeline_v6.agent_level. Levels: S see a doctor, L2 discuss soon,
    L1 worth watching, INFO neutral ovulation note, L0 nothing. Rotterdam domains counted once.
    axis2_allowed defaults to False because Axis 2 (watch ovulation) failed its K4 check."""
    reasons = []
    paused = hormonal_contraception or pregnant_or_breastfeeding
    if paused:
        reasons.append("cycle checks paused (hormonal contraception, pregnancy or breastfeeding)")
    if no_period_90 and not paused:
        return ("S", "You haven't logged a period for 3 months or more. If there's any chance you could be "
                     "pregnant, take a test. Either way, please see a doctor about it.", ["no period 90+ days"])
    early = years_since_menarche is not None
    diary_ok = not paused and not (early and years_since_menarche < 2)
    watch_ok = not paused and axis2_allowed and not (early and years_since_menarche < 2)
    o_diary = diary_ok and (axis1_pattern or axis1_infrequent)
    o_watch = watch_ok and axis2
    a = bool(model2_flag_cycle_neutral) if model2_flag_cycle_neutral is not None else bool(model2_flag)
    m = bool(scan_pcom)
    if o_diary:
        reasons.append("cycle lengths outside 21-35 days consistently, or fewer than 8 a year")
    if o_watch:
        reasons.append("no sign of ovulation in her last 2 answered cycles")
    if a:
        reasons.append("Model 2 flag from androgen-related signs")
    if m:
        reasons.append("polycystic ovaries on a scan she entered")
    domains = int(o_diary or o_watch) + int(a) + int(m)
    if domains >= 2:
        return ("L2", "What you've logged, taken together, is worth discussing with a doctor soon. It can have "
                      "several causes, and only a doctor can tell which.", reasons)
    if domains == 1 and o_watch and not (o_diary or a or m):
        return ("INFO", "In your last two cycles the watch didn't pick up the usual signs that ovulation "
                        "happened. This happens now and then in healthy cycles, and the watch can miss it.", reasons)
    if domains == 1:
        return ("L1", "Something in your logs is worth keeping an eye on. Keep logging and we'll look again "
                      "after your next cycles. If it continues, mention it to your GP.", reasons)
    return ("L0", "Nothing in your logs needs action right now. Keep logging.", reasons)
