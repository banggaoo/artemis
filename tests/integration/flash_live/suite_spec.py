"""Complex-UI suite: task definitions for the ~100-run Artemis Flash
on-device evaluation. Every task is verified by deterministic device
state (XML elements / element attributes / system queries) — never by
model self-report.

Each task:
  goal            — natural-language instruction handed to the model
  intent          — `am start -a` action (or main_activity for pkg launch)
  package         — force-stopped before each run
  oracle          — {type, params}: how the final tree/system is checked
  invariants      — declared goal predicates (Flash gate)
  seed            — optional pre-run ADB state actions
  timeout         — wall clock for the run
  hierarchy_backend — which a11y backend the run forces
"""

TASKS = {
    # ---- Digit entry on a 50-element lookalike pad -------------------------
    "dial_5550155": {
        "goal": "Dial 5550155 using the on-screen dial pad.",
        "intent": "android.intent.action.DIAL",
        "package": "com.google.android.dialer",
        "hierarchy_backend": "uiautomator",
        "timeout": 900,
        "invariants": [{
            "id": "dialed",
            "selector": {"resource_id": "com.google.android.dialer:id/digits",
                         "class": "android.widget.EditText"},
            "attribute": "text", "expected": "555-0155",
        }],
        "oracle": {"type": "text_equals", "resource_id": "com.google.android.dialer:id/digits",
                   "expected": "555-0155"},
        "seed": [{"type": "clear_app_data"}],
    },
    "dial_8675309": {
        "goal": "Dial 8675309 using the on-screen dial pad.",
        "intent": "android.intent.action.DIAL",
        "package": "com.google.android.dialer",
        "hierarchy_backend": "uiautomator",
        "timeout": 900,
        "invariants": [{
            "id": "dialed",
            "selector": {"resource_id": "com.google.android.dialer:id/digits",
                         "class": "android.widget.EditText"},
            "attribute": "text", "expected": "867-5309",
        }],
        "oracle": {"type": "text_equals", "resource_id": "com.google.android.dialer:id/digits",
                   "expected": "867-5309"},
        "seed": [{"type": "clear_app_data"}],
    },
    "dial_5551234": {
        "goal": "Dial 5551234 using the on-screen dial pad.",
        "intent": "android.intent.action.DIAL",
        "package": "com.google.android.dialer",
        "hierarchy_backend": "uiautomator",
        "timeout": 900,
        "invariants": [{
            "id": "dialed",
            "selector": {"resource_id": "com.google.android.dialer:id/digits",
                         "class": "android.widget.EditText"},
            "attribute": "text", "expected": "555-1234",
        }],
        "oracle": {"type": "text_equals", "resource_id": "com.google.android.dialer:id/digits",
                   "expected": "555-1234"},
        "seed": [{"type": "clear_app_data"}],
    },
    "dial_911": {
        # Short sequence: the model must stop at 3 digits, not over-dial.
        "goal": "Dial 911 using the on-screen dial pad.",
        "intent": "android.intent.action.DIAL",
        "package": "com.google.android.dialer",
        "hierarchy_backend": "uiautomator",
        "timeout": 900,
        "invariants": [{
            "id": "dialed",
            "selector": {"resource_id": "com.google.android.dialer:id/digits",
                         "class": "android.widget.EditText"},
            "attribute": "text", "expected": "911",
        }],
        "oracle": {"type": "text_equals", "resource_id": "com.google.android.dialer:id/digits",
                   "expected": "911"},
        "seed": [{"type": "clear_app_data"}],
    },

    # ---- Text input + dense suggestion list --------------------------------
    "sms_5550155": {
        "hierarchy_backend": "uiautomator",
        "goal": "Start a new text message conversation to the number 5550155.",
        "intent": "android.intent.action.VIEW",
        "intent_data": "sms:",
        "package": "com.google.android.apps.messaging",
        "timeout": 900,
        "invariants": [{
            "id": "recipient",
            "selector": {"class": "android.widget.TextView",
                         "text": "Texting with 555-0155 (SMS/MMS)"},
            "attribute": "text", "expected": "Texting with 555-0155 (SMS/MMS)",
        }],
        "oracle": {"type": "text_present", "text": "Texting with 555-0155"},
        "seed": [{"type": "clear_app_data"}],
    },
    "sms_1234567": {
        "hierarchy_backend": "uiautomator",
        "goal": "Start a new text message conversation to the number 1234567.",
        "intent": "android.intent.action.VIEW",
        "intent_data": "sms:",
        "package": "com.google.android.apps.messaging",
        "timeout": 900,
        "invariants": [{
            "id": "recipient",
            "selector": {"class": "android.widget.TextView",
                         "text": "Texting with 123-4567 (SMS/MMS)"},
            "attribute": "text", "expected": "Texting with 123-4567 (SMS/MMS)",
        }],
        "oracle": {"type": "text_present", "text": "Texting with 123-4567"},
        "seed": [{"type": "clear_app_data"}],
    },

    # ---- Dense scrollable navigation ---------------------------------------
    "nav_battery": {
        "goal": "Open Battery settings.",
        "intent": "android.settings.SETTINGS",
        "package": "com.android.settings",
        "timeout": 900,
        "invariants": [{
            "id": "battery_page",
            "selector": {"resource_id": "android:id/title",
                         "class": "android.widget.TextView",
                         "text": "Battery usage"},
            "attribute": "text", "expected": "Battery usage",
        }],
        "oracle": {"type": "text_present", "text": "Battery usage"},
        "seed": [],
    },
    "nav_sound": {
        "goal": "Open Sound & vibration settings.",
        "intent": "android.settings.SETTINGS",
        "package": "com.android.settings",
        "timeout": 900,
        "invariants": [{
            "id": "sound_page",
            "selector": {"class": "android.widget.TextView", "text": "Media volume"},
            "attribute": "text", "expected": "Media volume",
        }],
        "oracle": {"type": "text_present", "text": "Media volume"},
        "seed": [],
    },
    "nav_storage": {
        "goal": "Open Storage settings.",
        "intent": "android.settings.SETTINGS",
        "package": "com.android.settings",
        "timeout": 900,
        "invariants": [{
            "id": "storage_page",
            "selector": {"class": "android.widget.TextView", "text": "Free up space"},
            "attribute": "text", "expected": "Free up space",
        }],
        "oracle": {"type": "text_present", "text": "Free up space"},
        "seed": [],
    },
    "appinfo_chrome": {
        # Settings → Apps → See all apps → Chrome (a long alphabetized list).
        "goal": "Open the app info page for Chrome in Settings.",
        "intent": "android.settings.SETTINGS",
        "package": "com.android.settings",
        "timeout": 900,
        "invariants": [{
            "id": "chrome_info",
            "selector": {"class": "android.widget.Button", "text": "Force stop"},
            "attribute": "text", "expected": "Force stop",
        }],
        "oracle": {"type": "text_present", "text": "Force stop"},
        "seed": [],
    },

    # ---- State toggles on busy pages ----------------------------------------
    "battery_pct_on": {
        "goal": "Turn on 'Battery percentage' so it shows in the status bar.",
        "intent": "android.intent.action.POWER_USAGE_SUMMARY",
        "package": "com.android.settings",
        # The row's Switch is clickable=false/focusable=false (the row owns
        # the tap) — the a11y helper drops it from the tree entirely, so the
        # invariant and oracle can never see it there (verified live).
        "hierarchy_backend": "uiautomator",
        "timeout": 900,
        "invariants": [{
            "id": "battery_pct_switch",
            "selector": {"resource_id": "com.android.settings:id/switchWidget",
                         "class": "android.widget.Switch"},
            "attribute": "checked", "expected": True,
        }],
        # The Switch node is invisible to every live backend (see above) —
        # verify the Settings.System value it writes instead.
        "oracle": {"type": "system_setting", "namespace": "system",
                   "key": "status_bar_show_battery_percent", "expected": "1"},
        "seed": [{"type": "ensure_element_attr",
                  "intent": "android.intent.action.POWER_USAGE_SUMMARY",
                  "resource_id": "com.android.settings:id/switchWidget",
                  "attribute": "checked", "expected": "false"}],
    },
    "dark_off": {
        "goal": "Turn off Dark theme.",
        "intent": "android.settings.DISPLAY_SETTINGS",
        "package": "com.android.settings",
        "timeout": 900,
        "invariants": [{
            "id": "dark_theme_off",
            "selector": {"resource_id": "com.android.settings:id/switchWidget",
                         "class": "android.widget.Switch"},
            "attribute": "checked", "expected": False,
        }],
        "oracle": {"type": "night_mode", "expected": "Night mode: no"},
        "seed": [{"type": "uimode", "value": "yes"}],
    },
    "dark_on": {
        "goal": "Turn on Dark theme.",
        "intent": "android.settings.DISPLAY_SETTINGS",
        "package": "com.android.settings",
        "timeout": 900,
        "invariants": [{
            "id": "dark_theme_on",
            "selector": {"resource_id": "com.android.settings:id/switchWidget",
                         "class": "android.widget.Switch"},
            "attribute": "checked", "expected": True,
        }],
        "oracle": {"type": "night_mode", "expected": "Night mode: yes"},
        "seed": [{"type": "uimode", "value": "no"}],
    },

    # ---- Clock: existing alarms + hard creation -----------------------------
    "alarm_830_on": {
        "hierarchy_backend": "uiautomator",
        "goal": "Turn on the 8:30 AM alarm in the Clock app.",
        "intent": "main_activity",
        "main_activity": "com.google.android.deskclock/com.android.deskclock.DeskClock",
        "package": "com.google.android.deskclock",
        "timeout": 900,
        "invariants": [{
            "id": "alarm_830_switch",
            "selector": {"class": "android.widget.Switch", "text": "8:30 AM alarm"},
            "attribute": "checked", "expected": True,
        }],
        "oracle": {"type": "element_attr_text", "text": "8:30 AM alarm",
                   "attribute": "checked", "expected": "true"},
        "seed": [{"type": "ensure_element_attr",
                  "intent": "main_activity",
                  "main_activity": "com.google.android.deskclock/com.android.deskclock.DeskClock",
                  "text": "8:30 AM alarm", "attribute": "checked", "expected": "false"}],
    },
    "alarm_900_off": {
        "hierarchy_backend": "uiautomator",
        "goal": "Turn off the 9:00 AM alarm in the Clock app.",
        "intent": "main_activity",
        "main_activity": "com.google.android.deskclock/com.android.deskclock.DeskClock",
        "package": "com.google.android.deskclock",
        "timeout": 900,
        "invariants": [{
            "id": "alarm_900_switch",
            "selector": {"class": "android.widget.Switch", "text": "9:00 AM alarm"},
            "attribute": "checked", "expected": False,
        }],
        "oracle": {"type": "element_attr_text", "text": "9:00 AM alarm",
                   "attribute": "checked", "expected": "false"},
        "seed": [{"type": "ensure_element_attr",
                  "intent": "main_activity",
                  "main_activity": "com.google.android.deskclock/com.android.deskclock.DeskClock",
                  "text": "9:00 AM alarm", "attribute": "checked", "expected": "true"}],
    },
    "stopwatch_start": {
        "hierarchy_backend": "uiautomator",
        "goal": "Start the stopwatch in the Clock app.",
        "intent": "main_activity",
        "main_activity": "com.google.android.deskclock/com.android.deskclock.DeskClock",
        "package": "com.google.android.deskclock",
        "timeout": 900,
        "invariants": [{
            "id": "stopwatch_running",
            "selector": {"class": "android.widget.Button", "text": "Stop"},
            "attribute": "text", "expected": "Stop",
        }],
        "oracle": {"type": "text_present", "text": "Stop"},
        "seed": [{"type": "stopwatch_reset"}],
    },

    # ---- Search-driven navigation -------------------------------------------
    "settings_search_wallpaper": {
        "goal": "Use the Settings search bar to find and open 'Wallpaper & style' settings.",
        "intent": "android.settings.SETTINGS",
        "package": "com.android.settings",
        "timeout": 900,
        "invariants": [{
            "id": "wallpaper_page",
            "selector": {"class": "android.widget.TextView", "text": "Clock color & size"},
            "attribute": "text", "expected": "Clock color & size",
        }],
        "oracle": {"type": "text_present", "text": "Clock color & size"},
        "seed": [],
    },
}
