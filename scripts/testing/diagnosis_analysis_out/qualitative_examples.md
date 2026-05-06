# Qualitative examples — first-3-actions diff per bucket

## A. SFT_win (14) — RL lost what SFT had

### 10a730d5-d414-4b40-b479-684bed1ae522
**Instruction:** Considering I work late into the night and use Thunderbird frequently, I find that a full dark mode would be easier on my eyes during those hours. Can you help me enable a complete dark mode in Thunderbird?

**SFT** eval=1.0 steps=7  |  **ARPO** eval=0.0 steps=15

**SFT first 3:**
  1. Action: click(start_box='<|box_start|>(24,998)<|box_end|>')
  2. Action: click(start_box='<|box_start|>(135,288)<|box_end|>')
  3. Action: wait()

**ARPO first 3:**
  1. Action: click(start_box='<|box_start|>(61,14)<|box_end|>')
  2. Action: click(start_box='<|box_start|>(83,39)<|box_end|>')
  3. Action: wait()

### 276cc624-87ea-4f08-ab93-f770e3790175
**Instruction:** Please help me set the current user's line length for code wrapping to 50 characters in VS Code.

**SFT** eval=1.0 steps=11  |  **ARPO** eval=0.0 steps=15

**SFT first 3:**
  1. Action: click(start_box='<|box_start|>(30,1083)<|box_end|>')
  2. Action: wait()
  3. Action: hotkey(key='ctrl ,')

**ARPO first 3:**
  1. Action: click(start_box='<|box_start|>(444,1083)<|box_end|>')
  2. Action: click(start_box='<|box_start|>(388,905)<|box_end|>')
  3. Action: click(start_box='<|box_start|>(386,905)<|box_end|>')

### 3f28fe4f-5d9d-4994-a456-efd78cfae1a3
**Instruction:** Set up a plain text signature for my email account in Thunderbird. The first line is my name "Anonym" and the second line is my affiliation "XYZ Lab".

**SFT** eval=1.0 steps=6  |  **ARPO** eval=0.0 steps=15

**SFT first 3:**
  1. Action: click(start_box='<|box_start|>(25,996)<|box_end|>')
  2. Action: click(start_box='<|box_start|>(158,947)<|box_end|>')
  3. Action: click(start_box='<|box_start|>(457,668)<|box_end|>')

**ARPO first 3:**
  1. Action: click(start_box='<|box_start|>(1854,45)<|box_end|>')
  2. Action: click(start_box='<|box_start|>(17,1088)<|box_end|>')
  3. Action: click(start_box='<|box_start|>(17,1088)<|box_end|>')

### 53ad5833-3455-407b-bbc6-45b4c79ab8fb
**Instruction:** Please help me use VS Code to open the "project" in the "user" folder under "home".

**SFT** eval=1.0 steps=13  |  **ARPO** eval=0.0 steps=15

**SFT first 3:**
  1. Action: click(start_box='<|box_start|>(131,1078)<|box_end|>')
  2. Action: wait()
  3. Action: wait()

**ARPO first 3:**
  1. Action: click(start_box='<|box_start|>(117,1078)<|box_end|>')
  2. Action: wait()
  3. Action: wait()

### 6e99a1ad-07d2-4b66-a1ce-ece6d99c20a5
**Instruction:** Help me format column "spent" by keeping two decimal points.

**SFT** eval=1.0 steps=11  |  **ARPO** eval=0.0 steps=15

**SFT first 3:**
  1. Action: wait()
  2. Action: click(start_box='<|box_start|>(1236,683)<|box_end|>')
  3. Action: click(start_box='<|box_start|>(270,258)<|box_end|>')

**ARPO first 3:**
  1. Action: left_double(start_box='<|box_start|>(75,113)<|box_end|>')
  2. Action: left_double(start_box='<|box_start|>(133,1078)<|box_end|>')
  3. Action: wait()


## B. both_solve — ARPO >3 steps LONGER (0)

(Only 7/22 non-equal both_solve pairs had ARPO longer; most show ARPO is *shorter*.)


## B'. both_solve — ARPO MUCH SHORTER (>3 fewer steps) (4) — RL compressed the policy

### 2b94c692-6abb-48ae-ab0b-b3e8a19cb340 (SFT=15, ARPO=7, -8)
**Instruction:** Move the image to the right side on Slide 2.

**SFT first 3:**
  1. Action: left_double(start_box='<|box_start|>(77,309)<|box_end|>')
  2. Action: click(start_box='<|box_start|>(1265,473)<|box_end|>')
  3. Action: click(start_box='<|box_start|>(113,408)<|box_end|>')

**ARPO first 3:**
  1. Action: left_double(start_box='<|box_start|>(77,309)<|box_end|>')
  2. Action: click(start_box='<|box_start|>(115,419)<|box_end|>')
  3. Action: click(start_box='<|box_start|>(1265,473)<|box_end|>')

### a5bbbcd5-b398-4c91-83d4-55e1e31bbb81 (SFT=15, ARPO=7, -8)
**Instruction:** Enable VLC Minimal Interface in window mode so the bottom playback controls are hidden, and make sure the setting persists after restarting VLC. I often multitask on my computer, and the persistent toolbar in VLC is very distracting.

Task-family guidance:
- Use VLC/media-player controls and the Desktop file, not system settings or search loops.
- Typical VLC flow: open VLC, then Media -> Open File (or open the Desktop file directly in VLC).

**SFT first 3:**
  1. Action: click(start_box='<|box_start|>(359,384)<|box_end|>')
  2. Action: click(start_box='<|box_start|>(967,342)<|box_end|>')
  3. Action: click(start_box='<|box_start|>(1004,573)<|box_end|>')

**ARPO first 3:**
  1. Action: wait()
  2. Action: click(start_box='<|box_start|>(388,45)<|box_end|>')
  3. Action: click(start_box='<|box_start|>(967,346)<|box_end|>')

### f23acfd2-c485-4b7c-a1e7-d4303ddfe864 (SFT=15, ARPO=9, -6)
**Instruction:** Add a bullet point to the content of this slide.

**SFT first 3:**
  1. Action: click(start_box='<|box_start|>(90,283)<|box_end|>')
  2. Action: click(start_box='<|box_start|>(1265,476)<|box_end|>')
  3. Action: click(start_box='<|box_start|>(819,582)<|box_end|>')

**ARPO first 3:**
  1. Action: left_double(start_box='<|box_start|>(75,301)<|box_end|>')
  2. Action: click(start_box='<|box_start|>(1265,478)<|box_end|>')
  3. Action: click(start_box='<|box_start|>(869,582)<|box_end|>')

### 72b810ef-4156-4d09-8f08-a0cf57e7cefe (SFT=11, ARPO=7, -4)
**Instruction:** I am peer-reviewing my friend's course outline. I think the last paragraph is redundant so I want to add strike-through on words in the last paragraph. Can you do this for me?

**SFT first 3:**
  1. Action: click(start_box='<|box_start|>(88,303)<|box_end|>')
  2. Action: click(start_box='<|box_start|>(1224,683)<|box_end|>')
  3. Action: drag(start_box='<|box_start|>(647,648)<|box_end|>', end_box='<|box_start|>(1240,706)<|box_end|>')

**ARPO first 3:**
  1. Action: click(start_box='<|box_start|>(88,316)<|box_end|>')
  2. Action: click(start_box='<|box_start|>(1265,473)<|box_end|>')
  3. Action: drag(start_box='<|box_start|>(643,649)<|box_end|>', end_box='<|box_start|>(1240,706)<|box_end|>')


## C. ARPO_win (17) — RL improved over SFT

### 0e763496-b6bb-4508-a427-fad0b6c3e195
**Instruction:** Change the font to "Times New Roman" throughout the text.

**SFT** eval=0.0 steps=15  |  **ARPO** eval=1.0 steps=9

**SFT first 3:**
  1. Action: left_double(start_box='<|box_start|>(96,288)<|box_end|>')
  2. Action: wait()
  3. Action: click(start_box='<|box_start|>(1265,475)<|box_end|>')

**ARPO first 3:**
  1. Action: left_double(start_box='<|box_start|>(75,303)<|box_end|>')
  2. Action: click(start_box='<|box_start|>(1265,473)<|box_end|>')
  3. Action: hotkey(key='ctrl a')

### 2ad9387a-65d8-4e33-ad5b-7580065a27ca
**Instruction:** Can you make a new folder for me on the bookmarks bar in my internet browser? Let's call it 'Favorites.'

Task-family guidance:
- Browser search flow: click address bar/search field, type query, press Enter; avoid repeated new-tab/menu clicks if current tab works.

**SFT** eval=0.0 steps=15  |  **ARPO** eval=1.0 steps=13

**SFT first 3:**
  1. Action: click(start_box='<|box_start|>(966,126)<|box_end|>')
  2. Action: click(start_box='<|box_start|>(1748,124)<|box_end|>')
  3. Action: click(start_box='<|box_start|>(1752,124)<|box_end|>')

**ARPO first 3:**
  1. Action: click(start_box='<|box_start|>(954,103)<|box_end|>')
  2. Action: click(start_box='<|box_start|>(923,135)<|box_end|>')
  3. Action: click(start_box='<|box_start|>(923,135)<|box_end|>')

### 455d3c66-7dc6-4537-a39a-36d3e9119df7
**Instruction:** Could you help me export an Impress file to a .png image file and save it as res.png on the Desktop? Follow the default export setting is fine.

**SFT** eval=0.0 steps=15  |  **ARPO** eval=0.9588166750261486 steps=14

**SFT first 3:**
  1. Action: click(start_box='<|box_start|>(59,13)<|box_end|>')
  2. Action: click(start_box='<|box_start|>(1265,473)<|box_end|>')
  3. Action: click(start_box='<|box_start|>(1265,473)<|box_end|>')

**ARPO first 3:**
  1. Action: click(start_box='<|box_start|>(59,13)<|box_end|>')
  2. Action: click(start_box='<|box_start|>(1265,473)<|box_end|>')
  3. Action: click(start_box='<|box_start|>(1265,473)<|box_end|>')

### 510f64c8-9bcc-4be1-8d30-638705850618
**Instruction:** Could you start VS Code in folder ~/Desktop/project from the terminal?

**SFT** eval=0.0 steps=6  |  **ARPO** eval=1.0 steps=7

**SFT first 3:**
  1. Action: type(content='cd ~/Desktop/project')
  2. Action: hotkey(key='enter')
  3. Action: type(content='code')

**ARPO first 3:**
  1. Action: type(content='cd ~/Desktop/project')
  2. Action: hotkey(key='enter')
  3. Action: type(content='code .')

### 58d3eeeb-e9d0-499f-962e-fd0db2a744d8
**Instruction:** Based on the image above, translate the hidden audio conversation into French.

Task-family guidance:
- Use GIMP/image editor tools, not Ubuntu System Settings.
- If GIMP shows a blank canvas, use File -> Open to load the image before using Colors adjustments.
- For color tasks, prefer Colors menu actions (e.g., Color Balance/Saturation) over random toolbar/sidebar clicks.

**SFT** eval=0.0 steps=15  |  **ARPO** eval=1.0 steps=1

**SFT first 3:**
  1. Action: wait()
  2. Action: click(start_box='<|box_start|>(693,151)<|box_end|>')
  3. Action: click(start_box='<|box_start|>(693,151)<|box_end|>')

**ARPO first 3:**
  1. Action: call_user()
