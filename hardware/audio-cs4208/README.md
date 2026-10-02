# Subsystem: Audio Codec & High Definition Audio

* **Controllers:**
  * Intel 8-Series Lynx Point HD Audio Controller (`8086:8c20` on `00:1b.0`)
  * Intel Crystal Well HD Audio (`8086:0d0c` on `00:03.0`)
  * NVIDIA GK107 HDMI Audio (`10de:0e1b` on `01:00.1`)
* **Physical Audio Codec:** Cirrus Logic CS4208
* **Kernel Drivers:** `snd_hda_intel`, `snd_hda_codec_cirrus`

---

## 1. Hardware Architecture

The primary analog and optical audio path is managed by the Cirrus Logic CS4208 codec connected via Intel High Definition Audio (HDA) link to the PCH:

* **Internal Stereo Speakers:** Bi-amplified stereo speaker modules.
* **Dual Internal Digital Microphones:** MEMS microphone array for beamforming and ambient noise cancellation.
* **3.5mm Headphone Jack:** Combo port supporting:
  * Stereo analog headphone output.
  * TRRS microphone input (Apple headset pinout).
  * Optical S/PDIF TOSLink digital audio output (with physical optical switch detection).
