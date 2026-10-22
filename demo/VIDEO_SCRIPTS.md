# CycleSignal demo videos

Two videos, recorded from the same app (`streamlit run app/cyclesignal_app.py`).

Each video comes in two lengths: a **60-second promo** (like the team's example videos)
and a **full walkthrough** (about 3-4 minutes, for the assessment).

| | Video A: CycleSignal today | Video B: CycleSignal tomorrow |
|---|---|---|
| Purpose | Show what we built and that it runs on real models | Show where it goes with the right data |
| App mode | Prototype (today) | Vision (concept) |
| Length | 60 s promo, or about 3:30 walkthrough | 45 s promo, or about 2:30 walkthrough |
| Numbers | Real outputs from our code | Illustrative, labelled "concept" on screen |

Rule for both: never say "she has PCOS". Say "risk-awareness score", "pattern signal",
"worth discussing with a doctor".

---

## Promo cuts (60 seconds, in the style of the team's example videos)

The examples share one structure. Copy it:
- **Chapter bar** along the bottom, with the current chapter highlighted
  (Hook · Problem · Demo · Features · Get started).
- **Kinetic text:** one short phrase at a time, big, centred, revealed word by word.
- **Real screens in device mockups** (phone or laptop), slightly tilted, with a cursor click.
- **Logo end card** with the tagline.
- Brand: blush-to-lilac gradient, plum text (#34203a), raspberry accent (#a8336a),
  purple callouts (#6d5899), DM Sans, the pink and lilac circles from the title slide.

### Promo A: CycleSignal today (60 s)

| Time | Chapter | Big text on screen | Visual | Voiceover (optional) |
|---|---|---|---|---|
| 0-4 | Hook | "1 in 10 women" | Gradient + circles | "One in ten women has PCOS." |
| 4-8 | Hook | "Up to 70% don't know it." | Text alone | "Up to seventy percent don't know it." |
| 8-13 | Problem | "Scattered signs." → "Invisible between visits." | Icons: calendar, scales, skin, hair | "The signs are scattered, and invisible between GP visits." |
| 13-17 | Solution | **CycleSignal** logo + tagline | Logo reveal with circles | "CycleSignal brings the pattern together." |
| 17-22 | Demo | "Wear it." | Live clip: watch on wrist, then the Watch tab in a laptop mockup | "Wear your watch." |
| 22-27 | Demo | "Log it." | Phone mockup: Log tab, symptom toggles flipping on | "Log your period and a few symptoms." |
| 27-33 | Demo | "See the pattern." | Laptop: Score tab, zoom to **88.0 · Elevated pattern signal** | "See one clear score." |
| 33-40 | Demo | "Know why." | Zoom on the bar chart: skin darkening +12, weight gain +9, hair growth +8, irregular periods +7.5 | "And exactly what's driving it, for you." |
| 40-47 | Features | "Ask anything." | Assistant answer typing out, then the "Read more" links | "Ask questions. Get trusted reading." |
| 47-52 | Features | "Bring it to your GP." | Doctor summary, Download button click | "Take a summary to your GP." |
| 52-57 | Get started | "Earlier awareness. Not a diagnosis." | Logo + circles | "CycleSignal: earlier awareness, not a diagnosis." |
| 57-60 | Get started | Team names, module, university | End card | – |

### Promo B: CycleSignal tomorrow (45 s)

Show "Concept · illustrative data" in a corner the whole time.

| Time | Chapter | Big text on screen | Visual |
|---|---|---|---|
| 0-5 | Intro | "What if your cycle could speak up?" | Gradient + circles |
| 5-11 | Problem | "Today: data from different people." | Two separate data icons drifting apart |
| 11-17 | Vision | "Tomorrow: one woman, end to end." | Icons merge: watch + symptoms + diagnosis |
| 17-24 | Vision | "Nine months. One story." | Vision → Timeline in a laptop mockup |
| 24-30 | Vision | "Two of three. See your GP." | Rotterdam tracker tab |
| 30-36 | Vision | "Labs in. Confidence up." | Labs & scan tab |
| 36-41 | Vision | "Doctors close the loop." | Clinician view tab |
| 41-45 | Call to action | "CycleSignal" + "Help us build it with the right data." | Logo end card |

### How to build the promos (free tools)

1. **Record the screens:** 3-6 second clips of each tab. For phone-shaped clips, open the
   app in Chrome, press F12, click the phone icon (device toolbar), choose iPhone 14 Pro,
   and record that window.
2. **Canva** (free; easiest to match the examples): Create → Video (1920×1080, or
   1080×1920 for vertical). Search templates for "app promo" or "product launch".
   - Set the brand colours above and the DM Sans font.
   - Use the **Mockups** app to put your clips inside phone and laptop frames.
   - Animate text with **Rise** or **Typewriter**, word by word.
   - Make the chapter bar from 5 rounded rectangles; fill the current one with raspberry.
3. **CapCut** (alternative): keyframe a slow zoom and tilt on each screen clip, add text
   with the "Pop" animation, and add a soft beat from the music library.
4. **Export** 1080p MP4. Keep text on screen at least 1.5 seconds per phrase.

---

## Before recording

1. Browser at 1440×900, zoom 110%, light theme, other tabs closed. Hide the Streamlit
   toolbar with the menu (⋮) → Settings → Wide mode on.
2. Open the app once beforehand so Model 2 is trained and cached (no spinner on camera).
3. Record the screen with OBS (1080p, 30 fps). Record the voiceover separately in a quiet room.
4. Live footage on a phone: a teammate (as "Sarah") putting on a watch, going to bed, waking up,
   tapping a phone. Cover any brand logo.
5. Edit in CapCut or DaVinci Resolve. Add captions. Calm music, low volume.

---

## Video A: CycleSignal today (prototype, real models)

| # | Time | Picture | Do this in the app | Voiceover |
|---|---|---|---|---|
| A1 | 0:00-0:15 | Black title cards | – | "One in ten women has PCOS. Up to seventy percent don't know it, and diagnosis can take years." |
| A2 | 0:15-0:30 | Live: Sarah puts on the watch, sleeps, wakes | – | "Meet Sarah, a fictional user. She wears her watch as usual. Overnight it records her skin temperature and resting heart rate." |
| A3 | 0:30-0:50 | Screen | Demo user **Sarah, day 1** → tab **2 · Log** | "She logs her period with one tap, and once a month she answers a few quick questions: weight, whether her periods are regular, how long they last, hair growth, skin darkening, acne, hair loss, fast food and exercise. No blood tests needed." |
| A4 | 0:50-1:05 | Screen | Tab **3 · Score**. Zoom to 66.0 and the yellow "Low confidence" banner | "On day one her score is 66.0, 'increased pattern signal'. The app is honest: with no cycle history yet, it uses a cautious placeholder and says so." |
| A5 | 1:05-1:25 | Screen | Demo user **Sarah, after 3 months** → tab **1 · Watch** | "Three months later. Her cycles were 38, 44 and 41 days, all longer than the typical 21 to 35. In two of three cycles, her watch didn't pick up the temperature rise that usually follows ovulation." |
| A6 | 1:25-1:45 | Screen | Tab **3 · Score**. Point at 88.0, then the raspberry/purple bar | "Her score is now 88.0, 'elevated pattern signal'. It combines two models: 40% from her cycle pattern and 60% from her check-in answers, scored by a model trained on 541 clinical patients." |
| A7 | 1:45-2:05 | Screen | Same tab, scroll to the bar chart | "This chart shows why, for her specifically. Skin darkening added 12 points to her check-in score, weight gain 9, excess hair growth 8, and irregular periods 7.5." |
| A8 | 2:05-2:40 | Screen | Tab **4 · Assistant** → **Explain my result**. Let the text appear, scroll slowly | "The assistant explains the result in plain words: her cycle lengths, her biggest symptoms, that temperature is context only, what the app can't see, and what to do next: see a GP and ask about hormone, blood sugar and ultrasound tests." |
| A9 | 2:40-2:55 | Screen | Click **Is it my weight gain?**, then **Does my temperature matter?** | "She can ask follow-up questions. Weight gain raised her check-in score by about 9 points. Temperature isn't part of the score at all." |
| A10 | 2:55-3:10 | Screen | Tab **5 · What-if**. Turn off **Frequent fast food**, turn on **Regular exercise** | "The what-if view shows which answers move her result: 88.0 drops to 75.1. It shows how the model behaves; it isn't a treatment plan." |
| A11 | 3:10-3:20 | Screen | Tab **6 · Doctor summary** → **Download summary** | "She gets a one-page summary to bring to her doctor." |
| A12 | 3:20-3:30 | Screen | Demo user **Aoife (contrast)** → tab **3 · Score** | "For Aoife, with regular cycles and few symptoms, the score is 19.7: a low signal. A low score still can't rule PCOS out." |
| A13 | 3:30-3:45 | Title card | – | "Our check-in model scores 0.92 ROC-AUC on held-out patients and catches about eight in ten cases, but it still misses some, and the combined score isn't clinically validated yet. That's what Video B is about." |

**Important lines to keep, even if you cut for time:** A4 (honest about missing history),
A7 (her own drivers), A8 "temperature is context only", A13 (limits).

**If you have a Claude API key:** switch on **Agent uses Claude** before A8. The wording
will differ from the offline explainer each run, so record two or three takes and pick the best.

---

## Video B: CycleSignal tomorrow (vision, illustrative)

Keep the yellow "Concept, illustrative data only" banner visible in every screen shot.

| # | Time | Picture | Do this in the app | Voiceover |
|---|---|---|---|---|
| B1 | 0:00-0:15 | Title card: "What CycleSignal could become" | – | "Our prototype shows the idea works. Here's what it could do with the right data, from the right people." |
| B2 | 0:15-0:35 | Split screen: today's chart / text "same person: watch + symptoms + diagnosis" | – | "Today, our cycle data and our symptom data come from different people in different countries. The key step is data from the same women, followed for six to twelve months, with a confirmed diagnosis at the end." |
| B3 | 0:35-1:00 | Screen | Mode **Vision (concept)** → tab **Timeline** | "With that data, Sarah's nine months become one story. Her cycles get longer, and a validated ovulation detector, from her watch plus optional LH strips, shows ovulation stopping. The score becomes a calibrated likelihood with an honest uncertainty band." |
| B4 | 1:00-1:20 | Screen | Tab **Rotterdam tracker** | "The assistant tracks the three areas doctors use. Two of three show evidence, so it recommends a GP visit and prepares a referral summary." |
| B5 | 1:20-1:40 | Screen | Tab **Labs & scan** | "If Sarah has test results, she can add them from a photo of her GP letter, with her consent, for a more confident reading." |
| B6 | 1:40-2:00 | Screen | Tab **Clinician view** | "Her doctor sees the full picture. And with consent, the doctor's diagnosis goes back to the model as a label, so it keeps learning." |
| B7 | 2:00-2:25 | Screen | Scroll to **What has to be true to build this** | "To get there we need same-person data, a diverse cohort, an ovulation detector that passes validation against hormone tests, a clinical study, and medical-device approval." |
| B8 | 2:25-2:35 | Title card: team names, "CycleSignal: earlier awareness, not diagnosis" | – | "CycleSignal: earlier awareness, so fewer women wait years for answers." |

---

## Checklist before you export

- [ ] Every number spoken in Video A matches the screen (88.0, 66.0, 19.7, 12 / 9 / 8 / 7.5, 75.1, 0.92)
- [ ] Nobody says "she has PCOS" or "diagnose"
- [ ] Temperature is described as context, never as the cause
- [ ] Video B's concept banner is visible in every app shot
- [ ] Captions on, text readable on a phone screen
- [ ] Limitations stated (A13, B2, B7)
