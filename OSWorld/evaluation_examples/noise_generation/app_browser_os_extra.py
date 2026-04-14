"""
app_browser_os_extra.py — Extended APPLICATION / BROWSER / OS prompts.

30 additional interruption templates that are genuinely distinct from
anything in templates.py or variants.py. Each has unique content arrays,
unique bash structure, and targets a behavior not already covered.
"""

from __future__ import annotations


def _c(src: str) -> str:
    return src.strip()


# ============================================================================
# Additional application-internal prompts
# ============================================================================

def app_signin_required() -> str:
    """Modal styled as an app-internal 'Please sign in' prompt. Cost 1."""
    return _c(r"""
APPS=("Drive" "Mail" "Docs" "Notes" "Chat" "Cloud Sync");
a=${APPS[$((RANDOM % ${#APPS[@]}))]};
zenity --question --title "$a" --text "Your session has expired.\nSign in to continue working." --ok-label="Sign in" --cancel-label="Later" --width=360 >/dev/null 2>&1 &
""")


def app_trial_expired() -> str:
    """Modal: trial-expired / upgrade prompt. Cost 1."""
    return _c(r"""
APPS=("Pro Suite" "Advanced Tools" "Plus Edition" "Cloud Workspace");
a=${APPS[$((RANDOM % ${#APPS[@]}))]};
zenity --info --title "$a" --text "Your trial has ended.\nUpgrade to continue using premium features." --width=360 >/dev/null 2>&1 &
""")


def app_first_run_tutorial() -> str:
    """Modal styled as a welcome/tutorial overlay. Cost 1."""
    return _c(r"""
APPS=("Productivity Pro" "Notes Plus" "Mail Suite" "Workspace" "Writer");
a=${APPS[$((RANDOM % ${#APPS[@]}))]};
zenity --question --title "$a" --text "Welcome! Would you like a quick walkthrough?\nIt takes about 60 seconds." --ok-label="Yes, show me" --cancel-label="Skip" --width=380 >/dev/null 2>&1 &
""")


def app_tip_of_the_day() -> str:
    """Modal: app-internal 'Tip of the Day'. Cost 1."""
    return _c(r"""
TIPS=(
  "Press Ctrl+Shift+P to open the command palette."
  "Pin frequently-used files to the sidebar for quick access."
  "Use keyboard shortcuts to speed up navigation."
  "Enable auto-save in Preferences."
  "Try split view for side-by-side editing."
);
t=${TIPS[$((RANDOM % ${#TIPS[@]}))]};
zenity --info --title "Tip of the Day" --text "$t" --width=340 >/dev/null 2>&1 &
""")


def app_feedback_request() -> str:
    """Modal: 'Rate your experience'. Cost 1."""
    return _c(r"""
zenity --question --title "We'd love your feedback" --text "How would you rate your experience so far?\nA quick 30-second survey helps us improve." --ok-label="Give feedback" --cancel-label="Not now" --width=360 >/dev/null 2>&1 &
""")


def app_whats_new_overlay() -> str:
    """Modal: 'What's new in this version'. Cost 1."""
    return _c(r"""
V=(3.2 4.1 5.0 6.3 7.1); v=${V[$((RANDOM % ${#V[@]}))]};
F=(
  "Faster search • Dark mode • Improved sync"
  "New widgets • Bug fixes • Performance boost"
  "Collaboration features • Cloud backup"
  "Redesigned UI • Accessibility improvements"
);
f=${F[$((RANDOM % ${#F[@]}))]};
zenity --info --title "What's new in v$v" --text "$f" --width=360 >/dev/null 2>&1 &
""")


def app_license_key_prompt() -> str:
    """Modal: 'Enter your license key'. Cost 1."""
    return _c(r"""
zenity --entry --title "License Activation" --text "Enter your license key to unlock premium features:" --width=360 >/dev/null 2>&1 &
""")


def app_export_dialog() -> str:
    """Modal: 'Export as ...' file-chooser dialog. Cost 1."""
    return _c(r"""
F=("PDF" "DOCX" "ODT" "HTML" "Markdown" "Plain text");
f=${F[$((RANDOM % ${#F[@]}))]};
zenity --file-selection --save --title "Export as $f" --filename="/tmp/export_$RANDOM.$f" >/dev/null 2>&1 &
""")


# ============================================================================
# Additional browser prompts & overlays
# ============================================================================

def browser_autofill_suggestion() -> str:
    """Small prompt styled as Chrome's autofill suggestion bar. Cost 1."""
    return _c(r"""
F=("name" "email" "phone" "address" "credit card");
f=${F[$((RANDOM % ${#F[@]}))]};
zenity --question --title "Chrome" --text "Autofill detected saved $f.\nUse it to fill this form?" --ok-label="Use" --cancel-label="No thanks" --width=340 >/dev/null 2>&1 &
""")


def browser_certificate_warning() -> str:
    """Modal: 'Your connection is not private'. Cost 1."""
    return _c(r"""
SITES=("api.vendor.com" "beta.app.io" "staging.service.net");
s=${SITES[$((RANDOM % ${#SITES[@]}))]};
zenity --warning --title "Privacy error" --text "Attackers might be trying to steal your information from $s (for example, passwords, messages, or credit cards)." --width=420 >/dev/null 2>&1 &
""")


def browser_location_permission() -> str:
    """Modal: 'Allow location access' browser prompt. Cost 1."""
    return _c(r"""
SITES=("maps.app.com" "local.restaurant.io" "weather.info.net" "events.near.me");
s=${SITES[$((RANDOM % ${#SITES[@]}))]};
zenity --question --title "Google Chrome" --text "$s wants to know your location.\nAllow this site to access your location?" --ok-label="Allow" --cancel-label="Block" --width=360 >/dev/null 2>&1 &
""")


def browser_microphone_permission() -> str:
    """Modal: 'Allow microphone access'. Cost 1."""
    return _c(r"""
SITES=("meet.chat.com" "conference.team.io" "record.tool.net");
s=${SITES[$((RANDOM % ${#SITES[@]}))]};
zenity --question --title "Google Chrome" --text "$s wants to use your microphone.\nAllow this site to access your microphone?" --ok-label="Allow" --cancel-label="Block" --width=360 >/dev/null 2>&1 &
""")


def browser_camera_permission() -> str:
    """Modal: 'Allow camera access'. Cost 1."""
    return _c(r"""
SITES=("meet.chat.com" "video.tool.io" "call.service.net");
s=${SITES[$((RANDOM % ${#SITES[@]}))]};
zenity --question --title "Google Chrome" --text "$s wants to use your camera.\nAllow this site to access your camera?" --ok-label="Allow" --cancel-label="Block" --width=360 >/dev/null 2>&1 &
""")


def browser_notifications_permission() -> str:
    """Modal: 'Allow notifications from this site'. Cost 1."""
    return _c(r"""
SITES=("news.example.com" "updates.tool.io" "alerts.service.net");
s=${SITES[$((RANDOM % ${#SITES[@]}))]};
zenity --question --title "Google Chrome" --text "Allow $s to show notifications?\nYou will be notified about news and updates." --ok-label="Allow" --cancel-label="Block" --width=360 >/dev/null 2>&1 &
""")


def browser_clipboard_permission() -> str:
    """Modal: 'Allow clipboard read access'. Cost 1."""
    return _c(r"""
SITES=("editor.web.io" "docs.cloud.net" "app.collab.tools");
s=${SITES[$((RANDOM % ${#SITES[@]}))]};
zenity --question --title "Google Chrome" --text "$s wants to read from your clipboard.\nAllow this site to access your clipboard?" --ok-label="Allow" --cancel-label="Block" --width=360 >/dev/null 2>&1 &
""")


def browser_fullscreen_exit_hint() -> str:
    """Toast-style notice at top: 'Press Esc to exit full screen'. Cost 0."""
    return _c(r"""
( zenity --info --title "Full screen mode" --text "Press Esc to exit full screen" --width=260 >/dev/null 2>&1 ) & disown;
sleep 0.3;
wmctrl -r "Full screen mode" -e "0,760,20,400,60" 2>/dev/null;
wmctrl -r "Full screen mode" -b add,above 2>/dev/null;
(sleep 3; wmctrl -c "Full screen mode" 2>/dev/null) & disown
""")


def browser_install_extension_banner() -> str:
    """Banner: 'Install extension to get X feature'. Cost 1."""
    return _c(r"""
E=("Dark Reader" "Grammar Check" "Tab Groups Pro" "Quick Notes" "Password Helper");
e=${E[$((RANDOM % ${#E[@]}))]};
( zenity --question --title "Chrome Web Store" --text "Add \"$e\" to Chrome?\nThis extension can read data on all websites you visit." --ok-label="Add extension" --cancel-label="No thanks" --width=400 >/dev/null 2>&1 ) & disown
""")


def browser_not_responding_banner() -> str:
    """Banner: 'Page is unresponsive'. Cost 1."""
    return _c(r"""
SITES=("slow-app.io" "heavy-site.com" "laggy.page.net");
s=${SITES[$((RANDOM % ${#SITES[@]}))]};
zenity --warning --title "Page unresponsive" --text "$s is not responding.\nYou can wait or close the page." --width=360 >/dev/null 2>&1 &
""")


def browser_close_multiple_tabs() -> str:
    """Modal: 'Are you sure you want to close N tabs?'. Cost 1."""
    return _c(r"""
N=$((RANDOM % 12 + 3));
zenity --question --title "Close $N tabs?" --text "You have $N tabs open in this window.\nClose all of them?" --ok-label="Close all" --cancel-label="Cancel" --width=340 >/dev/null 2>&1 &
""")


def browser_clear_browsing_data() -> str:
    """Modal: 'Clear browsing data' confirmation. Cost 1."""
    return _c(r"""
zenity --question --title "Clear browsing data?" --text "This will clear history, cookies, and cache from the last hour.\nContinue?" --ok-label="Clear" --cancel-label="Cancel" --width=380 >/dev/null 2>&1 &
""")


def browser_reopen_closed_tabs() -> str:
    """Toast: 'Reopen X closed tabs?'. Cost 1."""
    return _c(r"""
N=$((RANDOM % 5 + 2));
( zenity --question --title "Chrome" --text "Reopen $N previously closed tabs?" --ok-label="Reopen" --cancel-label="Dismiss" --width=320 >/dev/null 2>&1 ) & disown
""")


# ============================================================================
# Additional OS-level prompts
# ============================================================================

def os_disk_almost_full() -> str:
    """Warning: 'Disk space is running low'. Cost 1."""
    return _c(r"""
P=$((RANDOM % 8 + 85));
zenity --warning --title "Low disk space" --text "Your home partition is $P%% full.\nConsider deleting unused files." --width=340 >/dev/null 2>&1 &
""")


def os_wifi_connection_dialog() -> str:
    """Modal: 'Wi-Fi network available' prompt. Cost 1."""
    return _c(r"""
N=(GuestWifi CafeFreeWiFi Office5G Home-2.4G Airport-WiFi FreePublicWifi);
n=${N[$((RANDOM % ${#N[@]}))]};
zenity --question --title "Wi-Fi" --text "Network \"$n\" is available.\nConnect?" --ok-label="Connect" --cancel-label="Ignore" --width=340 >/dev/null 2>&1 &
""")


def os_print_dialog_stuck() -> str:
    """Modal: 'Printer is offline or out of paper'. Cost 1."""
    return _c(r"""
P=("HP LaserJet Pro" "Canon MFP" "Brother DCP" "Epson WorkForce");
p=${P[$((RANDOM % ${#P[@]}))]};
zenity --error --title "Print error" --text "$p is offline.\nCheck power and connection." --width=340 >/dev/null 2>&1 &
""")


def os_software_updater_banner() -> str:
    """Banner: 'System updates will be installed tonight'. Cost 1."""
    return _c(r"""
H=$((RANDOM % 4 + 1));
zenity --info --title "Software Updater" --text "$H pending updates will be installed tonight at 02:00.\nRestart will be required." --width=340 >/dev/null 2>&1 &
""")


def os_unattended_upgrades_notice() -> str:
    """Banner: 'Unattended upgrades running'. Cost 0."""
    return _c(r"""
zenity --notification --window-icon=info --text "Updates — Installing security updates in the background" 2>/dev/null &
""")


def os_antivirus_scan_notice() -> str:
    """Notification: 'Antivirus scan complete'. Cost 0."""
    return _c(r"""
N=$((RANDOM % 5000 + 1000));
zenity --notification --window-icon=info --text "Security — Scan complete: $N files checked, 0 threats" 2>/dev/null &
""")


def os_firewall_block_notice() -> str:
    """Notification: 'Firewall blocked incoming connection'. Cost 0."""
    return _c(r"""
PORT=$((RANDOM % 40000 + 10000));
zenity --notification --window-icon=info --text "Firewall — Blocked connection attempt on port $PORT" 2>/dev/null &
""")


def os_storage_quota_approaching() -> str:
    """Banner: 'Cloud storage quota approaching'. Cost 1."""
    return _c(r"""
U=$((RANDOM % 15 + 80));
zenity --warning --title "Cloud Storage" --text "You are using $U%% of your 15 GB quota.\nConsider upgrading or freeing space." --width=340 >/dev/null 2>&1 &
""")


def os_mail_arrived_toast() -> str:
    """Toast: 'N new messages in inbox'. Cost 0."""
    return _c(r"""
N=$((RANDOM % 5 + 1));
F=(Alex Maya Tom Priya Sarah);
f=${F[$((RANDOM % ${#F[@]}))]};
zenity --notification --window-icon=info --text "Mail — $N new from $f" 2>/dev/null &
""")


def os_power_adapter_disconnected() -> str:
    """Toast: 'Power adapter disconnected'. Cost 0."""
    return _c(r"""
zenity --notification --window-icon=info --text "Power — Adapter disconnected, on battery" 2>/dev/null &
""")
