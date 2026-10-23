# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compact observation must expose explicit XML state for labeled controls.

An ON and an OFF labeled Switch used to render identical lines — the
``checked``/``enabled`` attributes were dropped between the XML and the
model-facing text (google/artemis#187). Absent or invalid attributes must
render as unknown, not implicitly OFF.
"""

from artemis.utils.visualization import format_minimal_list_with_elements


def _switch(**kw):
    node = {
        "class": "android.widget.Switch",
        "content-desc": "Dark theme",
        "resource-id": "com.android.settings:id/switchWidget",
        "checkable": "true",
        "checked": "false",
        "enabled": "true",
        "bounds": "[901,1661][1038,1787]",
    }
    node.update(kw)
    return node


def _render(nodes):
    return format_minimal_list_with_elements(nodes, 1080, 2400)


def test_on_and_off_switch_render_differently():
    on_text, _, _ = _render([_switch(checked="true")])
    off_text, _, _ = _render([_switch(checked="false")])
    assert on_text != off_text
    assert "checked:true" in on_text
    assert "checked:false" in off_text


def test_checked_and_enabled_rendered_distinctly():
    text, _, _ = _render([_switch(checked="true", enabled="true")])
    assert "checked:true" in text
    assert "enabled:true" in text


def test_missing_checked_renders_unknown():
    node = _switch()
    del node["checked"]
    text, _, _ = _render([node])
    assert "checked:?" in text
    assert "checked:false" not in text


def test_invalid_checked_renders_unknown():
    text, _, _ = _render([_switch(checked="yes")])
    assert "checked:?" in text


def test_disabled_checkable_control_shows_enabled_state():
    text, _, _ = _render([_switch(enabled="false")])
    assert "enabled:false" in text


def test_disabled_noncheckable_element_shows_enabled_false():
    node = {
        "class": "android.widget.TextView",
        "text": "Advanced",
        "enabled": "false",
        "bounds": "[0,0][500,100]",
    }
    text, _, _ = _render([node])
    assert "enabled:false" in text


def test_enabled_noncheckable_element_has_no_state_noise():
    node = {
        "class": "android.widget.TextView",
        "text": "General",
        "enabled": "true",
        "bounds": "[0,0][500,100]",
    }
    text, _, _ = _render([node])
    assert "checked" not in text
    assert "enabled" not in text


def test_state_carried_on_element_dict():
    _, elements, _ = _render([_switch(checked="true")])
    assert elements[0]["checked"] == "true"
    assert elements[0]["enabled"] == "true"


def test_identity_index_and_bounds_preserved():
    text, elements, labels = _render([_switch(checked="true")])
    assert "'Dark theme'" in text
    assert "[834,692][961,744]" in text
    assert elements[0]["index"] == 1
    assert elements[0]["resource_id"] == "com.android.settings:id/switchWidget"
    assert labels == ["1"]
