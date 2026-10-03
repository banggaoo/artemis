# iOS Simulator support

Artemis can run Flash and Pro tasks on iOS simulators on macOS with Xcode 27 or
newer. Select iOS explicitly: the existing Android defaults still apply on a
Mac. The first implementation supports the standalone CLI and embedded Python
SDK. The web console, Artemis daemon, remote `artemis-client`, and Artemis MCP
server currently use the Android device path.

The iOS driver uses tools included with Xcode and Artemis's existing Python MCP
dependency. It communicates with `xcrun mcpbridge` using an initialized MCP
session; it uses `xcrun simctl` for simulator and app lifecycle operations.
Appium, WebDriverAgent, and third-party simulator control utilities are not
required. Apple added native agent device interactions in Xcode 27; see the
[Xcode 27 release notes](https://developer.apple.com/documentation/xcode-release-notes/xcode-27-release-notes).

## Prepare Xcode

1. Install Xcode 27 or newer, complete its first-launch setup, and install an iOS
   simulator runtime. Confirm that `xcodebuild -version` reports the intended
   version and that `xcode-select -p` points to that Xcode installation. You can
   also set `DEVELOPER_DIR` for the Artemis process when multiple Xcode versions
   are installed.
2. Enable external agent access using Apple's
   [Xcode MCP access instructions](https://developer.apple.com/documentation/xcode/giving-external-agents-access-to-xcode)
   and review any Xcode approval prompts. For the workspace-independent MCP
   server introduced in Xcode 27, follow Apple's enablement instructions in the
   [release notes](https://developer.apple.com/documentation/xcode-release-notes/xcode-27-release-notes).
   Artemis does not enable the server or grant permissions automatically.
3. Check the toolchain and available simulators from the repository root:

   ```bash
   bash scripts/setup_ios_env.sh
   ```

   This check reads the Xcode version, tool locations, simulator inventory, and
   `xcrun mcp-server status`. It does not install dependencies, boot a simulator,
   or change access settings. A passing check confirms prerequisites; the first
   driver connection checks native MCP tool availability and device access.
4. Install Artemis's Python dependencies and configure a model provider using
   the normal project configuration:

   ```bash
   uv sync --dev
   cp .env.example .env
   ```

   Fill in the provider credentials in `.env` and select the desired models in
   `config/artemis.jsonc`. The one-click Android startup and dependency installer
   are not needed for this iOS workflow.

## Run a task

List the simulators to find the UDID for an available iOS device:

```bash
xcrun simctl list devices available
```

Run a standalone task on that simulator:

```bash
uv run artemis run "Open Settings and inspect the General screen" \
  --platform ios --standalone --device-serial <SIMULATOR-UDID> --profile flash
```

An explicitly selected simulator is booted if necessary, and the driver waits
for it to finish booting. To use a simulator that is already running, omit
`--device-serial` or pass `--device-serial booted`. Exactly one available iOS
simulator must be booted; with zero or multiple booted devices, choose a UDID.
The driver resolves `booted` once and pins every later command to that UDID.

Use `--profile pro` to run the planner, operator, and verification workflow on
the same driver. Prompts and app identifiers should refer to iOS apps and bundle
identifiers, such as `com.apple.Preferences`.

To install a simulator build before the task, add
`--app-path /absolute/path/MyApp.app`. The bundle's `Info.plist` must provide
`CFBundleIdentifier`.

### First-run Xcode approval

The very first time a new Python interpreter asks Xcode for device access,
Xcode requires user approval for that interpreter and, if one is supplied, the
selected project folder. Point Artemis at an existing Xcode project or
workspace so the request can be recorded:

```bash
uv run artemis run "Open Settings" --platform ios --standalone \
  --ios-workspace /absolute/path/MyApp.xcodeproj
```

If approval is still pending, the run stops with an "Xcode Approval Required"
panel instead of retrying. Approve the interpreter and the selected folder from
the Xcode MCP menu bar icon, choosing **Always Allow** there if offered and you
want later runs to skip the prompt; then rerun the task. Advanced users can instead
inspect pending request IDs with `xcrun mcp-server status` and approve only
those entries via `sudo xcrun mcp-server approve <REQUEST-ID> --always` from
their own terminal; the CLI path requires admin rights and is performed by the
user, never by Artemis.

Granted approvals persist across fresh bridge processes; there is no need to
keep a bridge alive. A different interpreter path or build, a different project
folder, or an expiring grant can require approval again. The Always/persistent
choice belongs to you and Xcode; Artemis requests only scoped approval for its
interpreter and the folder you select and never enables global access.

## Embedded Python SDK

Configure iOS through the embedded SDK's builder:

```python
from artemis.sdk import Agent
from artemis.sdk.builders import AgentConfigBuilder

config = (
    AgentConfigBuilder()
    .for_ios_simulator("<SIMULATOR-UDID>")
    .with_default_profile("flash")
    .build()
)
agent = Agent(config=config)
```

To run the first-run approval flow against a specific project, pass the
existing project or workspace through the builder:

```python
config = (
    AgentConfigBuilder()
    .for_ios_simulator(
        "<SIMULATOR-UDID>",
        workspace_path="/absolute/path/MyApp.xcodeproj",
    )
    .build()
)
```

`with_ios_workspace(path)` applies the same setting, and `None` clears it.
The path is used only if Xcode refuses the initial session with an approval
error; already-approved runs never open a workspace.

The generic builder also accepts
`for_device(DevicePlatform.IOS, "<SIMULATOR-UDID>")`, with `DevicePlatform`
imported from `artemis.context`. Supplying only a `device_serial` without an iOS
configuration retains the existing Android selection behavior.

A minimal run looks like:

```python
agent = Agent(config=config)
try:
    await agent.init()
    result = await agent.run_task(goal="Open Settings", profile="flash")
    screenshot = await agent.get_screenshot()
finally:
    await agent.clean()
```

`init` resolves the simulator without opening a native session. Public
`get_screenshot()` and `install_app()` calls acquire the device execution
lease and own a short-lived native session that is closed before they return,
so they work both right after `init` and between tasks. An app installation
requested inside `run_task` reuses the task's existing lease and session
instead of acquiring a second one.

## Supported operations and limits

| Operation | iOS implementation |
| --- | --- |
| Device selection and boot | `simctl` inventory, explicit UDID, `boot`, and `bootstatus` |
| Screenshot and accessibility hierarchy | Xcode native device-interaction MCP session |
| Tap, long press, and swipe | Native synthesized touch events |
| Text entry | Native keyboard synthesis with `clear_exist=false` |
| Enter, Home, Power, volume, and app switcher | Native keyboard and button synthesis |
| App install, launch, and terminate | `simctl` using simulator `.app` bundles and bundle identifiers |

The native hierarchy is normalized into the element tree used by Artemis's
perception and action tools. Screenshots and touch coordinates are kept in the
same coordinate space. Custom UI without accessible elements still relies on
visual targeting.

This driver targets iOS simulators. Physical iPhones and iPads, watchOS, tvOS,
and visionOS are outside this implementation. Install a simulator build of an
app; a device `.ipa` or Android `.apk` is not interchangeable with a simulator
`.app` bundle. The driver does not build Xcode projects.

iOS has no system Back button. Use the app's visible navigation controls.
The native driver supports `enter`, `home`, `power`, `volume_up`, `volume_down`,
and `app_switch`. The `erase_one_char` and `focus_and_clear_text` actions are
not exposed. The `back` and `delete` keys and automatic replacement of existing
text fail with an explicit unsupported-operation error. When calling the `input_text` action,
pass `clear_exist=false`; the default requests text replacement. In direct
driver or controller calls, the equivalent argument is `clear_existing=False`.
To replace text, clear the field using its visible UI first.
Android shell commands, Logcat, Android resource identifiers, Android package
discovery, and the Android Accessibility Helper are unavailable on iOS.
Platform-specific operations fail explicitly when unsupported.

Screen recording, video analysis, and video-based replay are not supported by
this initial iOS driver. Use screenshots and the task trace to review a run.

## Troubleshooting

- **Xcode version or tools are incorrect:** inspect `xcodebuild -version`,
  `xcode-select -p`, and any `DEVELOPER_DIR` override. Command Line Tools alone
  do not provide the full Xcode simulator and device-interaction environment.
- **MCP tool missing or access denied:** follow Apple's MCP access instructions,
  review approval prompts, and check `xcrun mcp-server status`. Xcode 27 release
  notes mention that some settings may require relaunching Xcode or restarting
  the Mac. If Xcode requests workspace approval before allowing a session, open
  and approve the intended workspace through Xcode's normal access flow.
- **CoreSimulator inventory fails:** run `xcrun simctl list devices available`
  in a local terminal. Sandboxed processes need access to the simulator service
  and its device data. Install a runtime and create a simulator if the iOS
  inventory is empty.
- **Multiple booted simulators:** supply `--device-serial` with the intended
  simulator's UDID.
- **Android daemon or device checks appear:** include `--platform ios
  --standalone`; iOS tasks currently run in the embedded process.

## Contributing and validation

Keep native MCP and subprocess interactions behind the iOS driver. Unit tests
should use fake MCP sessions and command runners so the normal deterministic
suite remains usable on Linux and without Xcode or model credentials. Run
`make test`, `make lint`, and `make typecheck` before submitting a change.

For live acceptance on a configured Mac, run the prerequisite script and an
explicit standalone CLI task. Verify that the chosen simulator receives taps,
swipes, and text, and that the screenshot and hierarchy describe the same
screen. Test an ambiguous `booted` selection and denied MCP access as well as a
successful session, and confirm that the session is released when the task
ends or is cancelled. Record the Xcode version, simulator runtime, UDID, and
observed results in the pull request; unit tests alone do not establish live
device compatibility.
