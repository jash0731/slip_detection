# Reviewed experiment protocol: pressure-conditioned LIF slip warning

This is a design to execute, not measured results. A designer and an independent challenger reviewed the protocol on 14 October 2026. The confirmed hardware is ESP32-WROOM-32, ADXL345, FSR, breadboards, jumper wires, LED, 10 kOhm resistors and USB data cable; a phone is available for video labels. The minimum contribution is a defined pressure-conditioned LIF network, independently labelled recordings, a fair software comparison, and an ESP32 demo retained by the college. It detects visible sliding in a manual gripper fixture; it does not establish incipient slip prediction or automatic grip tightening.

Execute in this order: build and inspect the circuit; run the pilot; collect development recordings; tune and freeze the models; collect the held-out live tests; replay and score; package the demo and report. The 126-trial research dataset excludes pilot trials and incorrect attempts, which must still be retained and documented.

## 1. Hypothesis and contribution

Test whether adding pressure-change excitation and limited inhibition to an acceleration-driven LIF SNN reduces false alarms during non-slip disturbances without materially reducing slip detection. Both networks use the same pressure contact gate. The change under investigation is the pressure-change connections, not merely using an FSR to detect contact.

Primary success target on untouched test trials: at least 30% relative reduction in squeeze/regrip plus loosen-without-slide negative-trial false alarms (24 test trials) and at most five percentage points decrease in overall slip recall. All-negative false-alarm rate is a secondary metric; isolated taps are a negative control, not a guaranteed beneficiary of pressure conditioning. Report absolute counts alongside percentages. Baseline zero false alarms makes relative reduction undefined; that outcome cannot establish the proposed advantage. Also inspect maintained-grip slip and squeeze-plus-slip separately so an aggregate improvement cannot conceal their failure. With 36 positive test trials, the five-percentage-point guard allows at most one fewer detected slide than baseline; two fewer means a 5.56-point decrease.

## 2. Build one reusable fixture

Use a simple manually operated two-jaw fixture made from available scrap wood, rigid cardboard, or existing clamp components. Secure one jaw to a base; squeeze the opposing jaw manually. Add a ruler or paper marks beside the object, and mark the jaw too. Relative object-to-jaw movement is the label, not overall fixture motion. Do not hold the accelerometer by hand or mount it on the sliding object.

Mount the ADXL345 rigidly to the sensor jaw, close to its contact pad. Put the FSR behind a thin protective pad, facing the object. Secure sensor wiring with tape so cable motion does not pull on the jaw. The fixture must let the object move tangentially while maintaining normal contact pressure. Use three safe household test objects with different contact surfaces, all compatible with the jaw: for example a rigid plastic strip, cardboard-covered block, and wooden strip. The conclusion will apply only to this fixture and tested objects.

Use every available electronic component: ESP32 for acquisition and inference; ADXL345 via I2C; FSR with one 10 kOhm resistor as a divider; LED with another 10 kOhm series resistor; breadboard/jumpers for connections; USB cable for power, flashing, and recording. Household fixture materials are passive construction, not additional sensing hardware. The 10 kOhm LED resistor may make the warning dim; test visibility and shade it if needed. Do not claim a bright indicator without checking it.

Use 3.3 V logic and power only after verifying the particular ADXL345 breakout. Divider topology: 3.3 V -> FSR -> ADC node -> 10 kOhm -> GND, so higher pressure should increase ADC readings. Choose the ESP32's configured ADC input range/attenuation and moderate grasp pressure so readings stay away from ADC clipping. A nominal 3.3 V divider does not guarantee the selected ADC setting measures its full output. Check baseline, contact, and squeeze values before testing. Keep Wi-Fi off. The user confirmed ESP32-WROOM-32: use exposed GPIO21 for SDA, GPIO22 for SCL, GPIO34/ADC1 for the FSR node, and GPIO25 for the LED through its series resistor; check the board pin labels. Check the breakout I2C mode/address straps and existing I2C pull-ups. If pull-ups are absent, available 10 kOhm resistors can serve as pull-ups with short wires and a conservative 100 kHz I2C clock; verify successful sampling before increasing speed.

## 3. Pilot: prove the sensors can see the task first

Record 15 short trials: three ordinary slides, three maintained-grip slides, three squeezes without motion, three taps without motion, and three stable holds. Keep enough pressure to maintain contact during slip trials. Read raw acceleration and ADC values into a CSV and plot them against time. Confirm that the ADC does not clip and that sliding repeatedly produces usable acceleration activity at this jaw mounting. Compare sliding with stable holding, not just with taps. Include a quick loosen-without-slide check before tuning pressure excitation.

Go forward only if the recordings show a plausible sliding signal and pressure-change direction. Reposition the accelerometer closer to the contact, secure wiring, or adjust the pad if needed; then repeat the pilot. If smooth sliding remains indistinguishable from holding, explicitly report the sensing limitation and narrow the project to tested vibration-producing slides. Do not compensate by inventing labels from the FSR or presenting impact-after-drop detection as slip detection.

## 4. Concrete algorithm, shared between software and firmware

Start with 200 Hz sampling and ADXL345 output rate, giving a 5 ms update. Read all three axes and the FSR each tick. Use compact CSV at 230400 baud if reliable with this board; otherwise buffer acquisition or reduce the logging format. Log sequence number, timestamp, three raw axes, FSR ADC, and output events. Check sequence gaps and actual tick spacing. Do not print debug messages inside the sampling loop.

Remove slowly changing acceleration using a roughly 100 ms running low-pass estimate on each axis. Convert the absolute residual to acceleration spikes using a threshold, with a short refractory period to avoid repeated spikes from a single sample. Smooth pressure lightly, e.g. 20 ms. Normalize ADC values using an unloaded reference and a moderate-contact reference recorded during development, not a Newton-force calibration. Generate positive and negative pressure-change spikes from change over approximately 25 ms. Thresholds and normalization stay fixed during the final test.

Baseline: three acceleration LIF neurons feed one output LIF neuron. FSR contact gating enables the network and resets state when contact is lost. Proposed: the same backbone and gate, plus pressure-decrease and pressure-increase LIF neurons. Decrease spikes provide small positive input to the output; increase spikes provide limited negative input. The output threshold remains reachable through acceleration alone, so maintained-grip slip is possible. Define inhibitory influence as brief pulses, with a cumulative budget capped at 0.25 of the output firing threshold in any rolling 500 ms window. Repeated pressure-increase spikes must obey that budget. This limits interference; it does not guarantee that weak slips remain detectable. Use the same cap for pressure excitation and allow it only when an acceleration neuron fired in the preceding 100 ms. These are starting structural limits to retain through the initial development/test protocol; whether they preserve weak-slip detection is still empirical. Pressure excitation must not itself generate slip alarms without recent acceleration activity. These explicit limits are part of the proposed method, not post-test adjustments.

Use a standard discrete LIF update: voltage decays each tick, weighted input spikes add or subtract evidence, and reaching threshold emits a spike and resets voltage. Publish the exact final time constants, weights, refractory periods, pressure limits, and gate thresholds in the report. Starting values are implementation candidates, not validated physical constants. Keep the network tiny; manually selected parameters and a small grid search are sufficient. No deep-learning training infrastructure is required.

## 5. Minimum recorded dataset: 126 trials

Use three objects, seven conditions, six repetitions each. Each trial lasts six seconds: grasp/settle for two seconds, apply the assigned event during the next two seconds, then hold until the end. The trial clock is driven by firmware; use a start signal and instructions to operators. Collect repetitions 1–2 in a development session, then lock parameters and collect repetitions 3–6 in a separate test session. Randomize object/condition order within each session. All samples from a trial stay together. This yields 42 development and 84 test trials, containing 36 positive and 48 negative test trials.

Conditions:

1. Stable hold, no slide: negative.
2. Tap the jaw once while the object remains stationary: negative.
3. Squeeze/regrip while preventing tangential motion: negative.
4. Reduce pressure but keep the object stationary and in contact: negative. This directly challenges pressure-decrease excitation.
5. Reduce pressure enough for a controlled visible slide while retaining contact: positive.
6. Maintained-grip slide: pull the object tangentially while keeping the manual squeeze approximately unchanged: positive. Record whether its FSR proxy is within the predeclared steadiness tolerance; do not call this verified constant force.
7. Pull tangentially while increasing squeeze pressure: positive; this challenges inhibition.

Ground-truth labels come from observation of object-to-jaw motion. Use a clear 2 mm reference mark to define a macroscopic slide. If a planned negative trial actually slides, label it positive/incorrectly executed and record the reason; do not silently count it negative. Log execution failures before looking at algorithm results. Pressure signals may verify the intended manipulation but must never decide whether sliding occurred. FSR ADC is an uncalibrated pressure/contact proxy, affected by contact area, mounting and drift; unchanged manual squeeze is not proof of constant physical force. State a tolerance for approximately steady FSR proxy after the pilot; if it fails, retain the independently labelled slide under a documented execution-failure category and repeat the intended condition. Keep both original and replacement records.

## 6. Software experiment: identical replay, fair tuning

Run the same recorded samples and initialization through Python implementations of baseline and proposed algorithms. Tune only on development trials. Freeze signal preprocessing and contact gating identically across models. Independently tune each model's available decision parameters with an equal cap, e.g. 30 candidate settings per model. Select the lowest false-alarm count among settings achieving at least 85% development recall; ties prefer better recall, then simpler parameters. If neither reaches 85%, revisit the pilot before collecting test data. This is fairer than deliberately leaving the baseline poorly tuned.

Test both models once on all test recordings. Also run a controlled ablation: proposed configuration with pressure-change connections set to zero while preserving everything else. The tuned baseline assesses practical improvement; the ablation isolates the added connections. Report both even if one comparison is less flattering.

Primary metrics: positive trials detected / positive trials; negative trials with any alarm / negative trials; per-condition counts. Start the scoring window after the two-second settling phase and end it before intentional final release. A positive detection requires a new alarm onset between independently annotated motion onset and 500 ms later (or the end of motion, whichever occurs first). An alarm before onset is an early/false alarm and does not count as detection, even if its LED hold overlaps sliding. Treat alarms within one camera frame of onset as timing-ambiguous; report that count separately. Merge repeated output spikes within 250 ms into one warning event; match at most one event per slide and count additional events separately. Hold the LED visibly for approximately 500 ms, but retain actual spike onset timestamps for scoring. Show a paired baseline/proposed outcome table; modest sample size supports a preliminary demonstration, not universal superiority.

Keep timing-ambiguous trials in the denominator. Report conservative recall using only unambiguous matches and an upper bound that also includes ambiguous matches. Apply the improvement target to the conservative result and show whether the conclusion changes at the upper bound. Report delay only for matched events, with the number of misses next to it. If recording or annotation is unusable, document that procedural failure before inspecting model output and retain the original attempt. Weak sensor signals or missed detections are never exclusion reasons.

## 7. Hardware experiment and timing

The user confirmed a phone is available. Film the jaw/object reference marks and LED together, preferably at 60 fps if available; otherwise record and report the actual frame rate. Use a distinctive LED flash sequence before each trial, with the first flash timestamp logged, to synchronize video and samples. Independently annotate the first sustained relative motion that subsequently reaches the 2 mm mark, plus motion end. Do this without inspecting model logs; crop or cover the warning LED region during motion annotation. The video is ground truth, not either sensor. Keep the LED visible in the original video for demonstration and synchronization. Report latency only for correctly matched alarms, rounded to frames and accompanied by at least frame-sized timing uncertainty; report misses separately. This is visible sliding latency, not incipient-slip lead time.

Run both locked models on ESP32 during the 84 test trials, using identical acquired samples and separate network states. The LED shows the proposed method; log event timestamps for both. Replay those raw samples offline and compare Python/firmware output events for each model. If events disagree, resolve arithmetic/reset/timestamp differences before accepting hardware verification. The same test session therefore establishes detection outcomes, software replay consistency and live embedded feasibility without a second dataset. Measure proposed-model inference duration separately from baseline and logging. After testing, deploy proposed-only firmware with USB power sufficient for operation; the phone and laptop are recording tools, not inference engines.

On ESP32 measure average/max processing duration per sample and number of missed 5 ms deadlines. Record firmware binary size and available free heap with the chosen framework; distinguish application state from whole-firmware memory. Success for deployment: no missed sampling deadlines in the tested sessions and reliable standalone warnings with USB power. No battery-energy or power-efficiency claims without power measurement.

## 8. Deliverables and honest outcome

Retain the wired fixture, mounted sensors, labelled connections, firmware, and a short startup/demo card. Provide raw CSVs, independent labels, split manifest, final parameter file, replay script, and result tables/plots. The report uses the requested sections: Abstract, Introduction, Literature Review, Proposed Methodology, Experimental Results, Conclusion, References.

If the method suppresses squeeze-plus-slip or pressure relaxation causes false alarms, that is a failed hypothesis or design revision. Revise only using development data, then acquire a new test session; never retune on the test and present it as untouched. A smaller demonstrated benefit is acceptable if its limits and counts are stated honestly. Publication remains future scope, conditional on results and a more complete novelty check.

## 9. Wiring and recording checklist

| Connection | ESP32-WROOM-32 development board |
|---|---|
| ADXL345 SDA / SDI | GPIO21 |
| ADXL345 SCL / SCLK | GPIO22 |
| ADXL345 power and ground | 3V3 and GND; verify breakout labels |
| FSR divider node | GPIO34: 3V3 -> FSR -> node -> 10 kOhm -> GND |
| External LED | GPIO25 -> 10 kOhm -> LED anode; LED cathode -> GND |
| Programming, power, logging | USB data cable |

Use the board's printed GPIO labels rather than counting header positions. On a breakout exposing CS and SDO, configure CS high for I2C and SDO low for address 0x53 unless the breakout already straps them. Confirm the address with an I2C scan. All grounds are common. Use a flat protective FSR pad that transfers normal pressure while avoiding direct tangential rubbing. Verify LED visibility before video collection; available 10 kOhm resistors can be connected in parallel for a lower effective LED resistance if necessary (for example five give 2 kOhm), retaining a series current-limiting resistance.

Suggested sample log columns:

`trial_id,seq,t_us,ax_raw,ay_raw,az_raw,fsr_adc,contact,baseline_spike,proposed_spike`

Store synchronization event timestamps separately or as marked log rows. Suggested independent label manifest:

`trial_id,object,condition,repetition,split,video_file,fps,sync_frame,sync_t_us,motion_onset_frame,motion_end_frame,visible_slide,procedure_valid,invalid_reason`

The first two seconds initialize shared filters and states; neither model is scored then. Reset both models identically before each trial and use identical timestamp handling in firmware and replay. Detect sample gaps and check fresh ADXL345 readings. Store raw recordings, configuration, labels and rejected attempts unchanged.

## 10. Blank report tables and figures

Each condition has 12 held-out trials, unless documented procedural replacements affect the achieved count. Fill these tables only after execution.

| Test condition | Planned test trials | Baseline outcome | Proposed outcome | Zero-pressure ablation |
|---|---:|---|---|---|
| Stable hold | 12 | False alarms: __/12 | __/12 | __/12 |
| Tap without slide | 12 | False alarms: __/12 | __/12 | __/12 |
| Squeeze/regrip without slide | 12 | False alarms: __/12 | __/12 | __/12 |
| Loosen without slide | 12 | False alarms: __/12 | __/12 | __/12 |
| Slide with reducing grip | 12 | Detected: __/12 | __/12 | __/12 |
| Slide while maintaining grip | 12 | Detected: __/12 | __/12 | __/12 |
| Slide while squeezing | 12 | Detected: __/12 | __/12 | __/12 |

| Summary measure | Baseline | Proposed |
|---|---|---|
| Primary pressure-disturbance false alarms | __/24 | __/24 |
| All negative-trial false alarms | __/48 | __/48 |
| Unambiguous slips detected | __/36 | __/36 |
| Timing-ambiguous slip trials | __/36 | __/36 |
| Early alarms during positive trials | __ | __ |
| Matched-event median delay, with frame uncertainty | __ | __ |
| Matched events contributing to delay | __ | __ |
| Firmware/replay agreement | __ | __ |
| Mean/max inference duration | __ | __ |
| Missed sample deadlines or sequence gaps | __ | __ |

Make three figures: (1) circuit/fixture photo and network diagram; (2) a clearly labelled held-out trace showing acceleration, FSR proxy, spikes and independent visible-motion interval; (3) condition-level false-alarm/detection counts. Show the primary relative reduction as `(baseline false alarms - proposed false alarms) / baseline false alarms`, when the baseline count is nonzero. Display all condition counts rather than selecting only successful objects.

Map into the requested report: Abstract states actual findings; Introduction explains the cheap-controller constraint; Literature Review distinguishes existing slip/SNN work from the pressure connections; Proposed Methodology describes wiring, LIF rules and trial protocol; Experimental Results includes these tables and timing/memory measurements; Conclusion states the supported benefit and failures; References includes primary papers and component documents.

For the retained demo, deploy the frozen proposed model as the default. Verify operation through three power cycles, including stable hold, squeeze without sliding, and a slide. USB supplies power; laptop computation and phone recording are unnecessary for the warning. Leave mounted and secured sensors, labelled wires, three test objects, firmware source/binary, final parameters and a one-page startup card describing grip, slide and LED behaviour. Report the result of this check rather than assuming it passes.

## 11. Primary references

- [Analog Devices ADXL345 datasheet](https://www.analog.com/media/en/technical-documentation/data-sheets/ADXL345.pdf): acceleration sensing, I2C configuration and sampling options.
- [Espressif GPIO documentation](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/gpio.html): classic ESP32 GPIO and ADC1 mapping.
- [Arduino-ESP32 ADC documentation](https://docs.espressif.com/projects/arduino-esp32/en/latest/api/adc.html): raw readings and attenuation-dependent range.
- [Interlink FSR integration guide](https://www.interlinkelectronics.com/downloads/integration-guides/fsr-400-series-integration-guide.pdf): mounting, response, drift and hysteresis caveats; the user's exact FSR model remains unknown.
- [Follmann et al., Touch and slippage detection in robotic hands with spiking neural networks](https://www.sciencedirect.com/science/article/pii/S0952197624011114): existing SNN slip detection; this project must distinguish its specific pressure-conditioning contribution.

Hardware documentation does not establish slip detectability in this fixture. Publication remains future scope after measured results and a deeper novelty review.
