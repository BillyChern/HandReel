/* Embedded 3D viewer (#viewer), one clip (figure data-seq).
   Layout, both modes: page controls on top (Play/Pause, Replay, a frame slider with one step per displayed frame,
   keyboard-operable, a text readout "frame NNN, T.TT s": the dataset's frame number and the time from the first
   frame shown; no source time stamp is shown or claimed),
   then the input frame in its own panel (left; first on phones) beside the 3D. The excerpt starts by itself when the
   viewer comes into view and loops, muted (the video has no audio track): after the last frame's own duration it
   starts again from the first frame with no added hold, until the visitor pauses, scrubs or switches the view. With reduced motion it neither starts by
   itself nor loops. Live, the meshes are drawn in the site's palette (PAPER, from the figure's data-obj / data-hand:
   reconstructed object dark purple, reconstructed hand light lilac); the pre-rendered videos and posters are rendered
   with the same palette (pipeline/viser_fallback.py); the recordings keep the colours they were recorded with.
   Live: the Viser playback client (viewer/client, same origin, pinned by sha256 in build_public.py) in an iframe plays
   viewer/recordings/<clip>.viser. The page drives the client through its own DOM and its viewer context, found by
   walking React's fiber tree from any client DOM node. checkClient() asserts every internal the page uses before the
   page takes over; if any is missing the page falls back to the video (it never runs half-driven). The client's
   playback bar and notices are hidden; time is read from its hidden slider input and set through its time field;
   play/pause is its own button. Every animation frame the page re-applies the camera limits (the client resets some,
   e.g. maxDistance, after its first renders), hides the recording's input inset and frame label (scene-tree visibility
   override; the page gives frame and time as text), applies the mesh opacity toggle, and, in the client's own
   animation-frame turn right after the client has drawn, paints the input frame of the frame group the client shows,
   so the input panel and the 3D change in the same rendered frame. The camera is constrained (azimuth and polar
   range, distance limits, no panning, slower rotation), WASD/QE camera keys are blocked, Reset view restores the start
   camera without reloading (play state and time are kept). On touch screens a layer over the client lets a vertical
   swipe scroll the page, turns a one-finger horizontal swipe into orbit and a two-finger pinch into zoom (clamped).
   Away from the start view the camera backs off so the billboard stays inside the canvas; a toggle switches
   between the live 3D view and the pre-rendered video (both ways); the viewer's notes sit under the controls.
   Loading: the client is given time while its page is still arriving or its recording download makes progress (a
   status line says it is loading); a recording answered with an error status, an error page, a client whose internals
   differ, a missing input index, or a download that stops for STALL_MS falls back to the video with a note that says
   why (an error status on the recording request is read from Resource Timing in the client's frame). Fallback (no WebGL, software WebGL unless the visitor asks for the viewer anyway, or a failure above): the
   pre-rendered video of the same frames, fetched whole and played from a blob URL (python http.server has no byte
   ranges). In this mode the picture the video presents is the one clock: every animation frame the page reads the
   frame mark of the picture (the excerpt time in its head strip, matched against viewer/fallback/<clip>.marks.json,
   pipeline/video_marks.py), draws that picture and shows the input frame and the read-out of the same frame in the
   same task. The browser's video clocks are not trusted for this: requestVideoFrameCallback's mediaTime ran two
   frames ahead of the picture in WebKit's Linux build and its callback one screen refresh ahead in Firefox, and
   currentTime is coarse in Firefox. Only when no mark can be read (no marks file, a decoder
   that draws nothing readable) does the page fall back to a clock: mediaTime where it agrees with currentTime, else
   currentTime, through frameAt, the one time-to-frame function of both modes. Without JS, or without the input
   index: the plain video with its own controls; the page controls and Reset view stay hidden.
   The client is self-contained: its code, style sheet, WebAssembly, environment map and the recording's textures are
   inside the client page and the recording, read through data: and blob: URLs, and it asks no other host for
   anything. A content policy that refuses inline script, WebAssembly, data: fetches or blob: images therefore stops
   it (the build's _headers file states the minimum policy). The page then does not leave a blank stage: a policy
   violation reported inside the client's frame, a client page whose script has not run BOOT_MS after it arrived, or a
   canvas that is still one flat colour, never rendered or still faded out BLANK_MS after the page took the client over each fall back to the video with
   one line that says why. Without WebGL the client is never started, also when ?viewer=webgl asks for it. The
   fallback video is played from a blob URL; where the policy refuses blob: media it is streamed from its path.
   Live field of view: the smallest that shows the whole textured plane (input frame and caption strip); the camera
   looks along the optical axis (data-look, so the plane faces the camera) and data-ext holds the plane's extent in
   that start camera (pipeline/viser_fallback.py report, page_data_look and page_data_ext). Wheel: plain scrolling
   scrolls the page, Ctrl + wheel and a middle-button drag zoom within the distance limits (both clamped by the page). Opened from a file (file://), the page never starts
   the viewer and fetches nothing: it shows the video and says how to serve the folder for the interactive viewer; the
   offline build embeds the input frame list (data-input). data-live="off" on the figure shows the video only (no
   viewer, no note, no Try anyway). */
(() => {
const boot = () => {
  // Sequence selector. Each example is a link "?clip=KEY#viewer" (works without JS, and from file://).
  // The default clip's figure is #viewer; every other served clip's figure waits, hidden, in a .vx-alt holder. Before
  // anything else reads the page, the requested clip's figure takes #viewer's place, so the rest of this file drives
  // exactly one figure, as before. An unknown key leaves the default clip.
  {
    const want = new URLSearchParams(location.search).get("clip");
    const cur = document.getElementById("viewer");
    if (want && cur && cur.dataset.seq !== want) {
      const alt = [...document.querySelectorAll(".vx-alt > figure.vx")].find((f) => f.dataset.seq === want);
      if (alt) {
        const holder = alt.parentElement;
        cur.removeAttribute("id");
        cur.replaceWith(alt);
        alt.id = "viewer";
        holder.appendChild(cur);
      }
    }
    const shown = document.getElementById("viewer");
    for (const a of document.querySelectorAll(".vx-seqs a[data-seq]")) {
      if (shown && a.dataset.seq === shown.dataset.seq) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    }
  }
  const root = document.getElementById("viewer");
  if (!root) return;
  document.documentElement.classList.remove("no-js");
  const key = root.dataset.seq;
  const stage = root.querySelector(".vx-stage");
  const incap = root.querySelector(".vx-incap");
  const outcap = root.querySelector(".vx-outcap");   // an HTML frame/time label under the 3D panel
  const video = stage.querySelector("video.vx-fallback");
  const box = root.querySelector(".vx-img");
  const note = root.querySelector(".vx-note");
  const sw = root.querySelector(".vx-sw");
  const status = root.querySelector(".vx-status");
  const tryBtn = root.querySelector(".vx-try");
  const againBtn = root.querySelector(".vx-again");     // after an automatic fallback: start the 3D viewer again
  const detailEl = root.querySelector(".vx-detail");    // ... and one line that says exactly what stopped it
  const detailText = root.querySelector(".vx-detail-text");
  const modeBtn = root.querySelector(".vx-mode");       // live: back to the video; video by choice: back to 3D
  const reset = root.querySelector(".vx-reset");
  const opacityBtn = root.querySelector(".vx-opacity");
  const labelFb = root.querySelector(".vx-label-fb");
  const labelLive = root.querySelector(".vx-label-live");
  const subFb = root.querySelector(".vx-sub-fb");          // one-line notes under the 3D (side view / orbited view)
  const subLive = root.querySelector(".vx-sub-live");
  // the live note ("Orbited, ...") shows only while the view is actually orbited (liveTick), never at the start view
  const setSub = (live) => { if (subFb) subFb.hidden = live; if (subLive) subLive.hidden = true; };
  const playBtn = root.querySelector(".vx-play");
  const replayBtn = root.querySelector(".vx-replay");
  const seek = root.querySelector(".vx-seek");
  const readout = root.querySelector(".vx-readout");
  const narrow = window.matchMedia("(max-width: 760px)");
  // One poster per viewport: the markup's poster is a <picture> behind the video (its media query, the same as
  // narrow's, fetched only the desktop or only the phone poster). Its choice becomes the video's poster attribute,
  // which the waits and view switches below read (same URL, so no second request), and the picture goes.
  {
    const pic = stage.querySelector(".vx-pic");
    if (pic) {
      const img = pic.querySelector("img");
      video.setAttribute("poster", (img && img.currentSrc) || `viewer/fallback/${key}${narrow.matches ? "-m" : ""}.jpg`);
      pic.remove();
    }
  }
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const STALL_MS = 15000;       // a recording download that makes no progress this long is given up
  const PAGE_CAP_MS = 180000;   // longest wait for the client's own page to arrive
  const MISMATCH_MS = 8000;     // a client internal still missing this long after its playback bar appeared: refuse it
  const FIT_MARGIN = 0.05;
  // radians, distance factors; fit: away from the start view the camera backs off (distance at least
  // start * (1 + fit * angle)) so that the input frame's billboard stays inside the canvas when orbited
  const ORBIT = { az: 0.45, polar: 0.3, near: 0.6, far: 1.6, speed: 0.5, fit: 1.2 };
  const NOTES = {
    failed: "The 3D viewer could not start here, so the page shows a pre-rendered video of the same frames.",
    stalled: "The 3D viewer stopped loading, so the page shows a pre-rendered video of the same frames.",
    unsupported: "The 3D viewer did not start as expected, so the page shows a pre-rendered video of the same frames.",
    nowebgl: "This browser has no WebGL, so the page shows a pre-rendered video of the same frames.",
    offline: "For the 3D viewer, run python3 -m http.server in this folder and open http://localhost:8000",
    blocked: "This server's content policy stops the 3D viewer, so the page shows a pre-rendered video of the same frames.",
    blank: "The 3D view did not appear. Showing a video of the same frames.",
  };
  const BOOT_MS = 5000;         // the client's page arrived but its own script has not run this long after: give up
  const BLANK_MS = 6000;        // the client runs but its canvas is still one flat colour this long after going live
  const LOOP_SETTLE_MS = 50;    // live only: after the seek back to the first frame the client settles this long before it plays
  // The site's palette (site.config.json "palette", the paper's method colour: reconstructed object dark purple,
  // reconstructed hand light lilac), written to the figure by pipeline/viewer_markup.py and applied to the live
  // meshes; pipeline/viser_fallback.py renders the videos and posters with the same two values.
  const hex = (v, d) => (/^#[0-9a-f]{6}$/i.test(v || "") ? parseInt(v.slice(1), 16) : d);
  const PAPER = { object: hex(root.dataset.obj, 0x6a3d9a), hand: hex(root.dataset.hand, 0xbca8d2) };

  /* ---- input frames: viewer/input/<clip>/index.json (frame ids, start times, files) ---- */
  let input = null;   // {frames: [{frame, t, src}], imgs, dur}
  let inputFailed = false;
  const ican = document.createElement("canvas");
  ican.className = "vx-icanvas";
  ican.setAttribute("aria-hidden", "true");
  box.appendChild(ican);
  // the frame whose time span [t_i, t_i+1) holds t (no look-ahead: the frame the video shows at t)
  const frameAt = (t) => {
    let i = 0;
    while (i + 1 < input.frames.length && input.frames[i + 1].t <= t + 1e-4) i += 1;
    return i;
  };
  const mid = (i) => {
    const f = input.frames;
    return (f[i].t + (i + 1 < f.length ? f[i + 1].t : input.dur)) / 2;
  };
  const label = (i) => {           // the caption strip's own numbering: dataset frame id and time from the first frame shown
    const t = input.frames[i].t.toFixed(2);
    const id = String(input.frames[i].frame).padStart(3, "0");
    return { text: `frame ${id}, ${t} s`, aria: `input frame ${id}, ${t} seconds` };
  };
  const paintInput = (i) => {
    if (!input) return;
    if (ican.width !== input.imgs[0].naturalWidth) {
      ican.width = input.imgs[0].naturalWidth;
      ican.height = input.imgs[0].naturalHeight;
    }
    ican.getContext("2d").drawImage(input.imgs[i], 0, 0, ican.width, ican.height);
  };
  let shown = -1;
  const showFrame = (i) => {        // readout, slider and input panel for displayed frame i
    if (!input || i < 0) return;
    paintInput(i);
    if (i === shown) return;
    shown = i;
    const l = label(i);
    readout.textContent = l.text;
    if (incap) incap.textContent = "Input " + l.text;   // the phone inset's own caption
    if (outcap) outcap.textContent = "Shown: " + l.text;
    seek.value = String(i);
    seek.setAttribute("aria-valuetext", (root.classList.contains("vx-approx") ? "about " : "") + l.aria);
  };
  // the frame is read from the picture (exact) or named by the video clock (approximate): the read-out shows the
  // word through the style sheet, the slider says it to a screen reader
  const setApprox = (on) => {
    if (root.classList.contains("vx-approx") === on) return;
    root.classList.toggle("vx-approx", on);
    if (shown >= 0) seek.setAttribute("aria-valuetext", (on ? "about " : "") + label(shown).aria);
  };

  /* ---- players: one interface for the client and the video ---- */
  let player = null;   // {time(), playing(), play(), pause(), seek(t)}
  let carry = null;    // the frame index the Video / 3D view switch keeps (null: start at the first frame)
  let carryPlay = false;   // ... and whether the excerpt was playing: the new view goes on playing from that frame
  // Taken when the visitor switches the view. While an earlier switch is still loading (its frame not yet applied),
  // that frame and play state are kept: the view that never came up has no frame of its own.
  const handOver = () => {
    if (carry !== null || !input) return;
    carry = shown >= 0 ? shown : 0;
    carryPlay = !!player && player.playing();
  };
  // Waiting for the first frame of a view: the stage shows the poster, or the picture held from the other view
  // (vx-held: the canvas keeps the last video frame, or a copy of the last 3D frame), with a one-line status, and is
  // never blank. The first frame of the new view replaces it in place.
  const waitFor = (text) => {
    root.classList.add("vx-wait");
    status.textContent = text;
    status.hidden = false;
  };
  const arrived = () => {
    if (!root.classList.contains("vx-wait")) return;
    root.classList.remove("vx-wait", "vx-held");
    noPoster();
    status.hidden = true;
  };
  // A loaded video shows its last decoded picture, not its poster. When the stage must show the poster again (a view
  // switch with no picture to hold), the poster is the stage's background and the video is hidden until its frame.
  const noPoster = () => { root.classList.remove("vx-poster"); stage.style.backgroundImage = ""; };
  const STATUS = { live: "Loading the 3D viewer ...", video: "Loading the video ..." };
  let mode = "none";
  // the frame on screen (shown: set by showFrame from the picture, in both modes) is the last one
  const atEnd = () => input && player && shown === input.frames.length - 1;
  // Loop: when playback reaches the last frame it shows that frame for its own duration, then starts again from the
  // first frame; nothing is added to the last frame. loopTimer is set meanwhile (and during the short settle after
  // the live seek), so playing() stays true and the
  // button keeps reading "Pause"; any pause, scrub or view switch cancels it.
  let loopTimer = null;
  const cancelLoop = () => { if (loopTimer) { clearTimeout(loopTimer); loopTimer = null; } };
  const restart = () => {
    loopTimer = null;
    if (!player || !input) return;
    player.seek(mode === "video" ? 0 : mid(0));
    if (mode === "live") loopTimer = setTimeout(() => { loopTimer = null; if (mode === "live" && player) { player.play(); syncButtons(); } }, LOOP_SETTLE_MS);
    else player.play();
    syncButtons();
  };
  const loopSoon = (ms) => { cancelLoop(); if (!reduced) loopTimer = setTimeout(restart, ms); };
  const syncButtons = () => {
    if (!player) return;
    playBtn.textContent = player.playing() ? "Pause" : "Play";
  };
  const doPlay = () => {
    if (!player || !input) return;
    cancelLoop();
    if (atEnd()) player.seek(mid(0));
    player.play();
  };
  // any use of the controls cancels the pending autoplay (it must not restart what the visitor paused)
  playBtn.addEventListener("click", () => { started = true; if (!player || !input) return; if (player.playing()) player.pause(); else doPlay(); syncButtons(); });
  replayBtn.addEventListener("click", () => { started = true; if (!player || !input) return; cancelLoop(); player.seek(mid(0)); player.play(); syncButtons(); });
  seek.addEventListener("input", () => {
    started = true;
    if (!player || !input) return;
    const k = parseInt(seek.value, 10);
    player.pause();
    player.seek(mid(k));
    // the slider says the frame the visitor chose at once; the read-out names the frame on screen (showFrame)
    seek.setAttribute("aria-valuetext", label(k).aria);
    syncButtons();
  });
  // Space toggles play anywhere in the viewer except on its own buttons and fields (they handle Space themselves).
  const onSpace = (e) => {
    if (e.code !== "Space" || !player || !input) return;
    started = true;
    const t = e.target;
    if (t && (t.tagName === "BUTTON" || t.tagName === "INPUT" || t.tagName === "SELECT" || t.isContentEditable)) return;
    e.preventDefault();
    if (player.playing()) player.pause(); else doPlay();
    syncButtons();
  };
  root.addEventListener("keydown", onSpace);
  // Space toggles playback only while focus is inside the viewer; elsewhere it scrolls the page.
  root.tabIndex = -1;

  // Start playing (and looping) when the viewer first comes into view (not with reduced motion).
  let started = false;
  const autoStart = () => {
    if (started || reduced || !player || !input || mode === "starting") return;
    started = true;
    player.seek(mid(0));
    if (mode === "live") setTimeout(() => { if (mode === "live" && !player.playing()) { player.play(); syncButtons(); } }, 500);
    else player.play();
    syncButtons();
  };
  // the visitor gets a moment to look at the viewer before the excerpt starts
  let visible = !("IntersectionObserver" in window);
  let seenTimer = null;
  if ("IntersectionObserver" in window) {
    new IntersectionObserver((es) => es.forEach((e) => {
      if (e.isIntersecting && !seenTimer && !visible) seenTimer = setTimeout(() => { visible = true; autoStart(); }, 600);
      else if (!e.isIntersecting && !visible && seenTimer) { clearTimeout(seenTimer); seenTimer = null; }
    }), { threshold: 0.25 }).observe(stage);
  }

  /* ---- WebGL check: hardware, software or none ---- */
  const SOFTWARE = /swiftshader|llvmpipe|softpipe|software|basic render/i;
  let glName = "";     // the renderer string of this page's own WebGL probe (shown in the detail line of a fallback)
  const webgl = (() => {
    try {
      const c = document.createElement("canvas");
      const gl = window.WebGLRenderingContext && (c.getContext("webgl2") || c.getContext("webgl"));
      if (!gl) return "none";
      const ext = gl.getExtension("WEBGL_debug_renderer_info");
      const name = String(gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER) || "");
      glName = name;
      const lose = gl.getExtension("WEBGL_lose_context");
      if (lose) lose.loseContext();
      return SOFTWARE.test(name) ? "software" : "hardware";
    } catch (e) {
      return "none";
    }
  })();
  const forced = new URLSearchParams(window.location.search).get("viewer") === "webgl";
  const offline = window.location.protocol === "file:";   // opened from a file: no iframe, no fetch
  const liveOff = root.dataset.live === "off";              // build switch: the video only, no viewer, no note

  /* ---- fallback: the pre-rendered video ---- */
  const vcan = document.createElement("canvas");
  vcan.className = "vx-vcanvas";
  vcan.setAttribute("aria-hidden", "true");
  stage.appendChild(vcan);
  let blobUrl = null;
  let blobReady = false;
  let wantPlay = false;       // play asked for before the video's data arrived (kept across the blob swap)
  let seekTarget = null;      // the time the page last asked for (slider, Replay, loop); cleared when playback starts
  const videoPlayer = {
    time: () => video.currentTime,
    playing: () => (!video.paused && !video.ended) || (wantPlay && !blobReady) || loopTimer !== null,
    play: () => { wantPlay = true; seekTarget = null; if (blobReady) video.play().catch(() => {}); },
    pause: () => { cancelLoop(); wantPlay = false; video.pause(); },
    // the visitor moved on: a frame still awaited after the wait is dropped, and a wait that is still running now
    // waits for the frame the visitor chose, with its full time
    seek: (t) => { seekTarget = t; late = null; if (want !== null) { want = frameAt(t); wantBy = 0; } video.currentTime = t; },
  };
  // Frame marks (pipeline/video_marks.py): which frame a picture of this video is, read from its pixels.
  let marksAll = root.dataset.marks ? JSON.parse(root.dataset.marks) : null;   // offline build: embedded
  let marks = null;           // for the loaded source: {x, y, w, h, pats: [Uint8Array], tol}
  const mcan = document.createElement("canvas");
  const mctx = mcan.getContext("2d", { willReadFrequently: true });
  const useMarks = () => {
    marks = null;
    const m = marksAll && marksAll.videos && marksAll.videos[(video.dataset.src || "").split("/").pop()];
    if (!m || !input || m.frames.length !== input.frames.length) return;
    if (m.size[0] !== video.videoWidth || m.size[1] !== video.videoHeight) return;
    const [x, y, w, h] = m.box;
    const pats = m.frames.map((f) => {
      const raw = atob(f.bits);
      const bits = new Uint8Array(w * h);
      for (let k = 0; k < w * h; k += 1) bits[k] = (raw.charCodeAt(k >> 3) >> (7 - (k & 7))) & 1;
      return bits;
    });
    mcan.width = w;
    mcan.height = h;
    marks = { x, y, w, h, pats, dark: m.dark_below_rgb_sum, tol: Math.floor(m.min_separation_bits / 3) };
  };
  // the index of the frame in a picture, or -1 if its mark cannot be read: by default the frame the video element
  // presents at this moment; given the canvas, the frame of the bitmap the canvas holds
  const readMark = (from = video) => {
    if (!marks) return -1;
    const { x, y, w, h, pats, dark, tol } = marks;
    let d = null;
    try {
      mctx.drawImage(from, x, y, w, h, 0, 0, w, h);      // the mark's box, pixel for pixel
      d = mctx.getImageData(0, 0, w, h).data;
    } catch (e) { marks = null; return -1; }
    let best = -1;
    let bestN = tol + 1;
    for (let i = 0; i < pats.length; i += 1) {
      const p = pats[i];
      let n = 0;
      for (let k = 0, j = 0; k < p.length && n < bestN; k += 1, j += 4) n += ((d[j] + d[j + 1] + d[j + 2] < dark) ? 1 : 0) ^ p[k];
      if (n < bestN) { bestN = n; best = i; }
    }
    return best;
  };
  let presented = null;   // media time of the frame the video last presented (requestVideoFrameCallback), a clock only
  // the clock, used only when the picture has no readable mark: mediaTime where it is plausible, else currentTime
  const videoClock = () => (presented !== null && Math.abs(presented - video.currentTime) < 0.15 ? presented : video.currentTime);
  let onCanvas = -1;      // the frame whose picture the canvas holds (-1: unknown, drawn by the clock)
  // The frame a view switch asked the video for, kept until a picture of it can be shown. The decoder can still
  // present an earlier frame for a few refreshes after "seeked": such a picture carries a readable mark of another
  // frame, and it is not copied, so the picture held from the other view stays on the canvas until the right one
  // arrives (or WANT_MS pass: then the picture that is there is shown, under its own label).
  let want = null;
  let wantBy = 0;
  const WANT_MS = 1500;
  // After that wait ends on another frame, the page goes on looking for the frame asked for (a paused video gets no
  // other repaint): until it shows, or LATE_WANT_MS pass, or the visitor moves the video. After that the picture on
  // screen stays, under its own label.
  let late = null;        // {i: the frame still awaited, until: performance.now() bound}
  const LATE_WANT_MS = 10000;
  const wanted = (m) => {
    if (m === want) return true;
    if (video.paused || video.ended) return false;
    const n = input.frames.length;       // playing: the excerpt has moved on from the frame asked for
    return (m - want + n) % n <= 3;
  };
  // Draws the picture the video presents and shows that frame's input image and read-out, in one task. With marks:
  // the picture is copied once and its mark is read from that copy, so the read-out and the input image always
  // belong to the bitmap on the canvas, whatever the video does between two calls.
  const scan = document.createElement("canvas");   // off-screen: the one copy of the video each call takes
  const sctx = scan.getContext("2d");
  const paintVideo = (force) => {
    if (late && performance.now() >= late.until) late = null;   // the bound holds while the video has no picture too
    if (mode !== "video" || !input || !root.classList.contains("vx-drawn") || video.readyState < 2) return;
    // the first picture after a view switch is the carried frame: nothing is drawn while the video still seeks to it
    if (video.seeking && root.classList.contains("vx-wait")) return;
    // The video is read ONCE per call, into the off-screen copy. Every decision below (is it the frame asked for,
    // is it new) and the picture that goes on screen use that copy, so the video cannot move between a check and
    // the copy that is shown.
    if (scan.width !== video.videoWidth || scan.height !== video.videoHeight) {
      scan.width = video.videoWidth;
      scan.height = video.videoHeight;
    }
    sctx.drawImage(video, 0, 0, scan.width, scan.height);
    const i = readMark(scan);
    if (want !== null) {
      if (root.classList.contains("vx-wait") && i >= 0 && !wanted(i)) {
        if (!wantBy) wantBy = performance.now() + WANT_MS;
        if (performance.now() < wantBy) return;       // the held picture stays on the canvas
        late = { i: want, until: performance.now() + LATE_WANT_MS };
      }
      want = null;
    }
    if (late && (i === late.i || !video.paused || performance.now() >= late.until)) late = null;
    if (vcan.width !== scan.width || vcan.height !== scan.height) {
      vcan.width = scan.width;
      vcan.height = scan.height;
      onCanvas = -1;
    }
    // nothing new to show: the copy is the frame whose verified bitmap the canvas holds
    if (!force && i >= 0 && i === onCanvas) { setApprox(false); showFrame(onCanvas); arrived(); return; }
    vcan.getContext("2d").drawImage(scan, 0, 0);
    if (i >= 0) { onCanvas = i; setApprox(false); showFrame(i); arrived(); return; }
    onCanvas = -1;                                   // no readable mark in the copy: the clock names the frame,
    setApprox(true);                    // and the read-out says that the frame is an estimate
    showFrame(frameAt(videoClock()));
    arrived();
  };
  // the stage takes the video's shape as soon as the video view is chosen (the markup carries both videos' sizes),
  // not when the video arrives, so the stage does not move when the first frame replaces the poster
  const videoShape = () => {
    const m = /^(\d+)x(\d+)$/.exec((narrow.matches ? video.dataset.sizeM : video.dataset.size) || "");
    if (m) stage.style.aspectRatio = `${m[1]} / ${m[2]}`;
  };
  const drawnMode = () => {
    if (mode !== "video" || !input || !video.videoWidth) return;
    stage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;   // no letterbox around the video
    video.controls = false;
    useMarks();
    // the frame kept across the view switch: the video seeks to it before its first picture is drawn
    if (carry !== null) { want = carry; wantBy = 0; late = null; video.currentTime = mid(carry); carry = null; carryPlay = false; }
    root.classList.add("vx-drawn");
    paintVideo(true);
  };
  const loadVideo = () => {
    const src = `viewer/fallback/${key}${narrow.matches ? "-m" : ""}.mp4`;
    if (video.dataset.src === src) return;
    if (offline) {                                   // file://: no fetch and no source swap (no aborted request)
      if (video.dataset.src) return;               // keep the browser's own first choice of source
      video.dataset.src = src;
      video.preload = "auto";
      const ready = () => { blobReady = true; if (wantPlay) video.play().catch(() => {}); };
      if (video.readyState >= 2) ready();
      else {
        video.addEventListener("loadeddata", ready, { once: true });
        if (video.networkState === HTMLMediaElement.NETWORK_EMPTY) video.load();
      }
      return;
    }
    const sources = Array.from(video.querySelectorAll("source"));
    const t = video.currentTime;
    videoShape();
    if (!video.paused && !video.ended) wantPlay = true;
    video.pause();
    sources.slice(1).forEach((e) => e.remove());   // one source, chosen here (the markup's media attribute: no JS)
    sources[0].removeAttribute("media");
    video.dataset.src = src;
    video.setAttribute("poster", src.replace(/\.mp4$/, ".jpg"));
    blobReady = false;
    const useSrc = (u) => {
      // a policy without "media-src blob:" refuses the blob URL (an error on the source element): stream the file
      sources[0].onerror = u === src ? null : () => { if (video.dataset.src === src) useSrc(src); };
      sources[0].setAttribute("src", u);
      video.load();
      video.addEventListener("loadeddata", () => {      // after the page's own loadeddata handler (drawnMode)
        blobReady = true;
        if (t && !video.seeking) video.currentTime = t;      // a source swap (the window was resized) keeps the time
        if (wantPlay) video.play().catch(() => {});
      }, { once: true });
    };
    sources[0].removeAttribute("src");              // fetched once, whole (no streamed request first)
    fetch(src)
      .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(String(r.status)))))
      .then((b) => {
        if (video.dataset.src !== src || mode !== "video") return;
        if (blobUrl) URL.revokeObjectURL(blobUrl);
        blobUrl = URL.createObjectURL(b);
        useSrc(blobUrl);
      })
      .catch(() => { if (video.dataset.src === src) useSrc(src); });   // stream from the served path instead
  };
  video.loop = false;
  video.addEventListener("loadeddata", () => drawnMode());
  // A pause settles on the frame that is on screen: the page seeks to that frame's middle (also when currentTime is
  // already there, so that a seek and its repaint always follow), which makes the video present exactly it. After a
  // pause, a seek or the end the page keeps painting for a moment (lively), because the decoder can deliver the
  // seeked picture a few screen refreshes after the event.
  let lively = 0;
  const settleLater = () => { lively = performance.now() + 700; };
  ["pause", "ended"].forEach((ev) => video.addEventListener(ev, () => {
    if (input && mode === "video" && !video.ended) {
      // a slider jump during playback pauses and seeks in one go: the pause must settle on the frame asked for, not
      // on the frame that was playing (the excerpt loops, so the slider is often moved while it plays)
      paintVideo();
      const i = seekTarget !== null ? frameAt(seekTarget) : onCanvas >= 0 ? onCanvas : frameAt(videoClock());
      video.currentTime = mid(i);
      syncButtons();
      return;
    }
    paintVideo();
    // the excerpt ended while it was playing (not paused by the visitor): the last frame has had its own duration,
    // so it starts again at once
    if (ev === "ended" && mode === "video" && input && wantPlay) loopSoon(0);
    syncButtons();
    settleLater();
  }));
  video.addEventListener("seeked", () => { paintVideo(); syncButtons(); settleLater(); });
  video.addEventListener("play", syncButtons);
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    const onFrame = (now, md) => { presented = md.mediaTime; video.requestVideoFrameCallback(onFrame); };
    video.requestVideoFrameCallback(onFrame);
  }
  // every animation frame while the video plays (and for a moment after a pause or a seek): the picture, the input
  // frame and the read-out are painted together from the picture's own frame
  const videoTick = () => {
    if (mode === "video" && ((!video.paused && !video.ended) || video.seeking || want !== null || late !== null || performance.now() < lively)) paintVideo();
    requestAnimationFrame(videoTick);
  };
  requestAnimationFrame(videoTick);

  const plainVideo = () => {          // no input frames: the plain video with its own controls, no page controls
    root.classList.add("no-input");
    tryBtn.hidden = true;               // the viewer needs the input frames too
    root.classList.remove("vx-drawn", "vx-wait", "vx-held");
    noPoster();
    status.hidden = true;
    video.controls = true;
  };
  /* ---- why the viewer gave way: one line of evidence for the visitor (and for whoever they send it to) ---- */
  const AUTO = { unsupported: 1, blank: 1, failed: 1, stalled: 1, blocked: 1 };   // fallbacks nobody chose
  const browserToken = () => {
    const ua = navigator.userAgent || "";
    const m = /(Edg|OPR|Firefox|FxiOS|CriOS|Chrome)\/[\d.]+/.exec(ua) || /Version\/[\d.]+(?= .*Safari)/.exec(ua);
    return m ? m[0].replace("Version/", "Safari ").replace("/", " ").replace(/^(\S+ \d+)\..*$/, "$1") : "unknown browser";
  };
  let startedAt = 0;      // when the current attempt to start the client began (wall clock)
  // "Viewer diagnostics: <what failed>, after 8.2 s, Chrome 131, WebGL <renderer>, tab visible, viewer 731x457, pixel ratio 2".
  // Shown under the note in small selectable text (after the retry button, which stays the prominent thing), mirrored to data-fallback / data-fallback-detail on the figure
  // and written once to the console. Nothing is sent anywhere.
  const showDetail = (why, what) => {
    if (!AUTO[why]) {
      if (detailEl) detailEl.hidden = true;
      delete root.dataset.fallback;
      delete root.dataset.fallbackDetail;
      return;
    }
    let size = "no frame";
    try { if (frame && frame.contentWindow) size = `${frame.contentWindow.innerWidth}x${frame.contentWindow.innerHeight}`; } catch (e) { size = "unreadable"; }
    const secs = startedAt ? ((Date.now() - startedAt) / 1000).toFixed(1) : "0.0";
    const text = `Viewer diagnostics: ${what || why}, after ${secs} s, ${browserToken()}, WebGL ${glName || webgl}, tab ${document.visibilityState}, ` +
      `viewer ${size}, pixel ratio ${window.devicePixelRatio}`;
    root.dataset.fallback = why;
    root.dataset.fallbackDetail = text;
    if (detailEl && detailText) { detailText.textContent = text; detailEl.hidden = false; }
    console.info(`[viewer] ${why}. ${text}`);
  };
  // Time that counts against the client: only while this tab is visible and the client's frame is getting animation
  // frames. A hidden or throttled tab runs timers but paints nothing; the client cannot finish starting there (its
  // canvas is only set up once it has been measured, which needs a painted frame), so that time must not use up the
  // budgets below. activeClock(win) returns a function that, called repeatedly, returns the active milliseconds so far.
  const activeClock = () => {
    let total = 0;
    let last = Date.now();
    let seen = -1;
    let frames = 0;
    let win = null;
    const count = () => { frames += 1; if (win) { try { win.requestAnimationFrame(count); } catch (e) { win = null; } } };
    const read = (w) => {
      if (w && w !== win) { win = w; try { w.requestAnimationFrame(count); } catch (e) { win = null; } }
      const now = Date.now();
      if (document.visibilityState === "visible" && frames !== seen && seen >= 0) total += Math.min(now - last, 1000);
      last = now;
      seen = frames;
      return total;
    };
    read.reset = () => { total = 0; last = Date.now(); seen = frames; };
    return read;
  };
  let clocks = [];        // the budgets of the current attempt; all start again when the tab's visibility changes
  document.addEventListener("visibilitychange", () => clocks.forEach((c) => c.reset()));
  let lateWatch = null;   // after "unsupported": the client stays mounted and is checked again (lateStart)
  const LATE_MS = 60000;
  const fallBack = (why, what, keepFrame) => {
    // a viewer that was playing when it had to give way (its view did not appear, or the policy refused a texture) hands a
    // playing excerpt over to the video
    const resume = carry !== null ? carryPlay : mode === "live" && !!player && player.playing();
    cancelLoop();
    if (watch) clearTimeout(watch);
    watch = null;
    if (lateWatch) clearInterval(lateWatch);
    lateWatch = null;
    showDetail(why, what);                 // before the frame goes: the line holds the client's size
    if (frame && !keepFrame) { frame.remove(); frame = null; }
    root.classList.toggle("vx-late", !!(frame && keepFrame));   // the kept client stays mounted, unseen, behind the video
    live = null;
    mode = "video";
    player = videoPlayer;
    root.classList.remove("is-live", "vx-drawn");
    stage.style.aspectRatio = "";
    videoShape();
    video.controls = false;
    // No picture was handed over (the frame asked for stays in carry for the seek). A loaded video's canvas can still
    // hold the frame it last showed, read from its mark: that picture stays, under its own label. Otherwise the canvas
    // is emptied and the poster shows, which is the first frame; a loaded video is hidden behind it.
    if (!root.classList.contains("vx-held") && input) {
      shown = -1;
      if (video.videoWidth && vcan.width > 0 && onCanvas >= 0) {
        root.classList.add("vx-held");
        setApprox(false);
        showFrame(onCanvas);
      } else {
        if (vcan.width > 0) vcan.getContext("2d").clearRect(0, 0, vcan.width, vcan.height);
        onCanvas = -1;
        const poster = video.getAttribute("poster");
        if (video.videoWidth && poster) { stage.style.backgroundImage = `url("${poster}")`; root.classList.add("vx-poster"); }
        setApprox(false);
        showFrame(0);
      }
    }
    waitFor(STATUS.video);
    opacityBtn.disabled = false;
    labelFb.hidden = false;
    labelLive.hidden = true;
    setSub(false);
    reset.hidden = true;
    opacityBtn.hidden = true;
    note.hidden = why === "software" || why === "off" || why === "chosen";
    if (NOTES[why]) note.textContent = NOTES[why];
    if (againBtn) againBtn.hidden = !AUTO[why] || offline || liveOff || webgl === "none";
    sw.hidden = why !== "software" || offline || liveOff;
    if (modeBtn) {                       // the way back to the 3D view after choosing the video
      modeBtn.hidden = why !== "chosen";
      modeBtn.textContent = "Interactive 3D";
    }
    loadVideo();
    if (inputFailed) plainVideo();
    else drawnMode();
    if (resume) player.play();
    syncButtons();
    if (visible) autoStart();
  };
  narrow.addEventListener("change", () => { if (mode === "video") loadVideo(); });

  /* ---- live: the Viser client ---- */
  let frame = null;
  let watch = null;
  let live = null;     // {win, doc, ctx, time field, hidden time input, play button, limits, ...}
  let seeThrough = false;
  const fovFor = () => {
    const [l, rt, top, bot] = root.dataset.ext.split(",").map(parseFloat);
    const r = stage.getBoundingClientRect();
    const h = Math.max(1, r.height);
    const t = Math.max(-top, bot, Math.max(-l, rt) / (r.width / h)) / (1 - FIT_MARGIN);
    return 2 * Math.atan(t);
  };
  const findContext = (doc) => {
    let el = null;
    let k = null;
    for (const e of doc.querySelectorAll("*")) {
      k = Object.keys(e).find((x) => x.startsWith("__reactFiber$"));
      if (k) { el = e; break; }
    }
    if (!el) return null;
    let f = el[k];
    while (f.return) f = f.return;
    const st = [f];
    while (st.length) {
      const x = st.pop();
      const v = x.memoizedProps && x.memoizedProps.value;
      if (v && v.mutable && v.mutable.current && "cameraControl" in v.mutable.current && v.sceneTreeActions) return v;
      if (x.child) st.push(x.child);
      if (x.sibling) st.push(x.sibling);
    }
    return null;
  };
  // Every client internal the page relies on; null if all are there, else the first one missing.
  const checkClient = (doc, ctx) => {
    const fn = (o, name) => o && typeof o[name] === "function";
    const slider = doc.querySelector(".mantine-Slider-root");
    const bar = slider && slider.closest(".mantine-Paper-root");
    if (!bar) return "playback bar";
    if (!bar.querySelector("input.mantine-NumberInput-input")) return "time field";
    if (!Array.from(bar.querySelectorAll('input[type="hidden"]')).some((x) => /^[0-9.eE+-]+$/.test(x.value))) return "time value";
    const button = bar.querySelector("button");
    if (!button || !button.querySelector("svg")) return "play button";
    if (!ctx) return "viewer context";
    const m = ctx.mutable.current;
    const c = m.cameraControl;
    for (const p of ["azimuthAngle", "polarAngle", "distance", "maxDistance", "mouseButtons", "touches"]) {
      if (!c || !(p in c)) return `camera ${p}`;
    }
    if (!fn(c, "rotate") || !fn(c, "dollyTo")) return "camera rotate";
    if (!fn(m, "resetCameraPose") || !fn(m, "requestRender") || typeof m.nodeRefFromName !== "object") return "viewer refs";
    if (!m.scene || typeof m.scene !== "object" || !m.canvas || typeof m.canvas.getContext !== "function") return "scene";
    if (!fn(ctx.sceneTreeActions, "updateNodeAttributes") || !ctx.useSceneTree || !fn(ctx.useSceneTree, "getAll")) return "scene tree";
    return null;
  };
  const limitsOf = (c) => {
    const az = c.azimuthAngle;
    const pol = c.polarAngle;
    const d = c.distance;
    return withBase({ az, pol, d }, { minAzimuthAngle: az - ORBIT.az, maxAzimuthAngle: az + ORBIT.az, minPolarAngle: Math.max(0.05, pol - ORBIT.polar),
      maxPolarAngle: Math.min(Math.PI - 0.05, pol + ORBIT.polar), minDistance: d * ORBIT.near, maxDistance: d * ORBIT.far,
      azimuthRotateSpeed: ORBIT.speed, polarRotateSpeed: ORBIT.speed, dollySpeed: 0.2 });
  };
  const withBase = (base, lim) => Object.defineProperty(lim, "base", { value: base, enumerable: false });
  // the nearest the camera may come at its current angle (the billboard stays in view when orbited)
  const minAt = (c, lim) => {
    const off = Math.max(Math.abs(c.azimuthAngle - lim.base.az), Math.abs(c.polarAngle - lim.base.pol));
    const ramp = Math.min(1, off / 0.1);   // at the start view the zoom limit holds; from 0.1 rad on, the full back-off
    const want = ORBIT.near + (1 - ORBIT.near + ORBIT.fit * off) * ramp;
    return Math.min(lim.maxDistance, Math.max(lim.minDistance, lim.base.d * want));
  };
  const clampD = (c, lim, d) => Math.min(lim.maxDistance, Math.max(minAt(c, lim), d));
  const applyLimits = (c, lim) => {
    for (const [k, v] of Object.entries(lim)) if (c[k] !== v) c[k] = v;
    const mn = minAt(c, lim);
    if (c.minDistance !== mn) c.minDistance = mn;
    if (c.distance > lim.maxDistance + 1e-6 || c.distance < mn - 1e-6) c.dollyTo(clampD(c, lim, c.distance), false);
    if (c.mouseButtons.right !== 0) c.mouseButtons.right = 0;   // camera-controls ACTION.NONE: no panning
    if (c.mouseButtons.middle !== 0) c.mouseButtons.middle = 0; // middle-drag dolly is the page's own (clamped) below
    if (c.touches.two !== 1024) c.touches.two = 1024;           // ACTION.TOUCH_DOLLY: pinch zooms, no two-finger pan
    if (c.touches.three !== 0) c.touches.three = 0;
  };
  const liveSetup = (win, doc) => {
    if (!doc.querySelector(".mantine-Slider-root")) return "notyet";
    if (!input) return inputFailed ? "noinput" : "notyet";
    const ctx = findContext(doc);
    const missing = checkClient(doc, ctx);
    if (missing) return `mismatch: ${missing}`;
    // the recording's frame groups arrive as it is decoded: wait for all of them (the stall rule still applies)
    const names = ctx.useSceneTree.getAll();
    if (input.frames.some((f) => { const g = `/f${String(f.frame).padStart(3, "0")}`; return !names[g] || !ctx.mutable.current.nodeRefFromName[g]; })) return "notyet";
    const bar = doc.querySelector(".mantine-Slider-root").closest(".mantine-Paper-root");
    const style = doc.createElement("style");
    style.textContent = '[class*="mantine-Notification"],[class*="mantine-Notifications"]{display:none!important}';
    doc.head.appendChild(style);
    bar.style.display = "none";
    const timeField = bar.querySelector("input.mantine-NumberInput-input");
    const hiddenTime = Array.from(bar.querySelectorAll('input[type="hidden"]')).find((x) => /^[0-9.eE+-]+$/.test(x.value));
    const button = bar.querySelector("button");
    const c = ctx.mutable.current.cameraControl;
    const limits = limitsOf(c);
    applyLimits(c, limits);
    // wheel: plain scrolling scrolls the page; Ctrl + wheel (and a trackpad pinch) zooms, clamped before the next draw
    win.addEventListener("wheel", (e) => {
      e.preventDefault();
      e.stopImmediatePropagation();
      if (!(e.ctrlKey || e.metaKey)) { window.scrollBy(0, e.deltaY); return; }
      const lim = live.limits;
      const want = c.distance * Math.exp(Math.max(-0.5, Math.min(0.5, e.deltaY * 0.002)));
      c.dollyTo(clampD(c, lim, want), false);
      ctx.mutable.current.requestRender();
    }, { capture: true, passive: false });
    // middle-button drag dollies through the page, clamped before the next draw like Ctrl + wheel: the client's own
    // drag dolly eased past the distance limit for a few frames before the per-frame clamp caught it
    let dragY = null;
    const dolly = (dy) => {
      const lim = live.limits;
      const want = c.distance * Math.exp(Math.max(-0.5, Math.min(0.5, -dy * 0.004)));   // drag up: out (as the client)
      c.dollyTo(clampD(c, lim, want), false);
      ctx.mutable.current.requestRender();
    };
    win.addEventListener("pointerdown", (e) => {
      if (e.button !== 1) return;
      e.preventDefault();
      e.stopImmediatePropagation();
      dragY = e.clientY;
    }, true);
    win.addEventListener("pointermove", (e) => {
      if (dragY === null) return;
      if (!(e.buttons & 4)) { dragY = null; return; }
      e.stopImmediatePropagation();
      dolly(e.clientY - dragY);
      dragY = e.clientY;
    }, true);
    win.addEventListener("pointerup", (e) => { if (e.button === 1 && dragY !== null) { dragY = null; e.stopImmediatePropagation(); } }, true);
    win.addEventListener("mousedown", (e) => { if (e.button === 1) e.preventDefault(); }, true);   // no autoscroll
    // keys: Space goes to the page's toggle; WASD/QE (camera truck and dolly) would leave the scene
    win.addEventListener("keydown", (e) => {
      if (e.code === "Space") { e.preventDefault(); e.stopImmediatePropagation(); if (player.playing()) player.pause(); else doPlay(); syncButtons(); }
      else if (/^Key[WASDQE]$/.test(e.code)) e.stopImmediatePropagation();
    }, true);
    // The client's play state follows React renders, so the page keeps the wanted state and liveTick clicks the
    // client's button until the two agree; setting the time pauses the client (its own rule).
    live = { win, doc, ctx, timeField, hiddenTime, button, limits, want: false, clicked: 0, seekTo: null,
      blank: activeClock(), probed: 0, draws: false, shows: false };
    clocks.push(live.blank);
    live.blank(win);
    const scene = ctx.mutable.current.scene;
    const afterRender = scene.onAfterRender;
    scene.onAfterRender = function onAfterRender(renderer) {   // the canvas, read right after the client drew it
      if (live && !live.draws && renderer && renderer.getRenderTarget() === null) live.draws = drawn(renderer.getContext());
      // a copy of the frame just drawn, asked for by the switch to the video (the drawing buffer is only readable now)
      if (live && live.snap && renderer && renderer.getRenderTarget() === null) {
        const snap = live.snap;
        live.snap = null;
        snap(renderer.domElement);
      }
      if (afterRender) return afterRender.apply(this, arguments);
      return undefined;
    };
    const setTime = (t) => {
      const set = Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, "value").set;
      set.call(live.timeField, t.toFixed(3));
      live.timeField.dispatchEvent(new win.Event("input", { bubbles: true }));
    };
    live.setTime = setTime;
    const clientPlaying = () => !!live.button.querySelector('svg[class*="player-pause"]');
    player = {
      time: () => parseFloat(live.hiddenTime.value) || 0,
      playing: () => live.want || loopTimer !== null,
      clientPlaying,
      play: () => { live.want = true; live.seekTo = null; },
      pause: () => { cancelLoop(); live.want = false; },
      seek: (t) => {
        live.want = false;
        live.seekTo = t;                 // liveTick sets it again if the client played on before it paused
        setTime(t);
      },
    };
    mode = "live";
    opacityBtn.disabled = false;
    root.classList.add("vx-drawn");
    player.pause();
    player.seek(mid(carry !== null ? carry : 0));   // Video -> 3D view keeps the frame
    // ... and goes on playing if the excerpt was playing (the client settles on the frame first, as at autoplay);
    // loopTimer is set meanwhile, so the button reads "Pause" and a pause or a scrub cancels it
    if (carry !== null && carryPlay && !reduced) {
      cancelLoop();
      loopTimer = setTimeout(() => { loopTimer = null; if (mode === "live" && player) { player.play(); syncButtons(); } }, 500);
    }
    carry = null;
    carryPlay = false;
    syncButtons();
    if (visible) autoStart();
    return "ok";
  };
  // meshes see-through (alignment against the real hand and object) or solid, in the paper's colours: a mesh
  // material's own colour is set (the geometry is the recording's, untouched); joints and bones drawn with per-vertex
  // colours keep the recording's colour
  const tint = () => {
    const m = live.ctx.mutable.current;
    const want = seeThrough ? 0.45 : 1;
    for (const [name, o] of Object.entries(m.nodeRefFromName)) {
      if (!o || !/^\/f\d{3}\/(object|hand_\w+)$/.test(name)) continue;
      const hex = /\/object$/.test(name) ? PAPER.object : PAPER.hand;
      o.traverse((x) => {
        if (!x.material) return;
        const c = x.material.color;
        if (x.isMesh && c && typeof c.getHex === "function" && !x.material.vertexColors && c.getHex() !== hex) {
          c.setHex(hex);
          x.material.needsUpdate = true;
          m.requestRender();
        }
        if (x.material.opacity === want && x.material.transparent === seeThrough) return;
        x.material.transparent = seeThrough;
        x.material.opacity = want;
        x.material.needsUpdate = true;
        m.requestRender();
      });
    }
  };
  // per animation frame while live: limits, hidden nodes, opacity, hold at the end, then paint after the client drew
  const liveTick = () => {
    if (mode !== "live" || !live) return;
    if (policyHit) { fallBack("blocked", `content policy: ${policyHit}`); return; }   // refused after the page took over (a texture, a late fetch)
    applyLimits(live.ctx.mutable.current.cameraControl, live.limits);
    if (subLive) {
      const cc = live.ctx.mutable.current.cameraControl, base = live.limits.base;
      const orbited = Math.max(Math.abs(cc.azimuthAngle - base.az), Math.abs(cc.polarAngle - base.pol)) > 0.03;
      if (subLive.hidden === orbited) subLive.hidden = !orbited;
    }
    const all = live.ctx.useSceneTree.getAll();
    for (const name of Object.keys(all)) {
      if (/\/(input|label)$/.test(name) && all[name].overrideVisibility !== false) {
        live.ctx.sceneTreeActions.updateNodeAttributes(name, { overrideVisibility: false });
      }
    }
    tint();
    if (player.clientPlaying() && player.time() >= input.frames[input.frames.length - 1].t) {
      const last = input.frames.length - 1;
      const reached = Math.min(player.time(), input.dur);   // how far into the last frame the client already is
      player.seek(mid(last));                        // stay on the last frame (setting the time pauses the client)
      loopSoon(Math.round((input.dur - reached) * 1000));   // for the rest of its own duration, then start again
      syncButtons();
    }
    const now = Date.now();
    if (player.clientPlaying() !== live.want && now - live.clicked > 200) {
      live.clicked = now;
      live.button.click();
    }
    // A seek made while the client plays (the excerpt loops, so the slider is often moved during playback) can land
    // late: the client pauses itself when its time is set, the page's own pausing click can then start it again, and
    // it plays on for a moment before the next click stops it. Once the client is paused and no click is pending,
    // the time is set again, until it holds.
    if (live.seekTo !== null && !live.want && !player.clientPlaying() && now - live.clicked > 300) {
      if (Math.abs(player.time() - live.seekTo) > 1e-3) live.setTime(live.seekTo);
      else live.seekTo = null;
    }
    // Until the canvas has been seen to draw (liveSetup's onAfterRender hook), keep asking the client for renders; a
    // canvas still flat, or never rendered, BLANK_MS after the page took over falls back to the video.
    // The canvas must also be shown: the client keeps it nearly transparent until its environment map has loaded,
    // so a client that cannot read its own data renders into a canvas nobody sees.
    if (!live.shows) {
      if (live.draws && displayed(live.ctx.mutable.current.canvas)) live.shows = true;   // the wait ends below, once the frame is painted
      else if (live.blank(live.win) > BLANK_MS) {
        fallBack(policyHit ? "blocked" : "blank", policyHit ? `content policy: ${policyHit}` : live.draws ? "client canvas drawn but not shown" : "client canvas not drawn");
        return;
      }
      else if (now - live.probed > 300) { live.probed = now; live.ctx.mutable.current.requestRender(); }
    }
    live.win.requestAnimationFrame(() => {
      if (mode !== "live" || !live) return;
      const refs = live.ctx.mutable.current.nodeRefFromName;
      const i = input.frames.findIndex((f) => {
        const o = refs[`/f${String(f.frame).padStart(3, "0")}`];
        return o && o.visible;
      });
      showFrame(i);
      // the first 3D frame is on screen (the client drew it and shows its canvas) with its input frame: the poster goes
      if (live.shows && i >= 0 && live.seekTo === null) arrived();
      syncButtons();
    });
    requestAnimationFrame(liveTick);
  };
  // The client's canvas shows something: right after the client has rendered its scene to the canvas (the scene's
  // onAfterRender; the drawing buffer is not kept between frames, so it cannot be read later), a 5 x 4 grid of its
  // pixels holds more than one colour. A canvas that never draws (textures or the client's own data refused by a
  // content policy, a context that renders nothing) stays one flat colour, or the scene is never rendered at all.
  const drawn = (gl) => {
    try {
      const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight, px = new Uint8Array(4);
      let first = null;
      for (let i = 1; i < 6; i += 1) {
        for (let j = 1; j < 5; j += 1) {
          gl.readPixels(Math.floor((w * i) / 6), Math.floor((h * j) / 5), 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
          const v = (px[0] << 24) | (px[1] << 16) | (px[2] << 8) | px[3];
          if (first === null) first = v;
          else if (v !== first) return true;
        }
      }
    } catch (e) { return false; }
    return false;
  };
  // The element is displayed and not faded out: the product of the opacities up its tree is above one half.
  const displayed = (el) => {
    if (!el || !el.isConnected || !el.getClientRects().length) return false;
    let alpha = 1;
    for (let e = el; e && e.nodeType === 1; e = e.parentElement) {
      const cs = e.ownerDocument.defaultView.getComputedStyle(e);
      if (cs.visibility === "hidden" || cs.display === "none") return false;
      alpha *= parseFloat(cs.opacity);
    }
    return alpha > 0.5;
  };
  // A content-policy violation inside the client's frame. Two refusals leave the client working and are ignored: its
  // WebAssembly glue probes Function() once, and its own tab icon is a data: image.
  let policyHit = null;
  const onViolation = (e) => {
    const what = e.effectiveDirective || e.violatedDirective || "";
    if (e.blockedURI === "eval" || (/^img-src/.test(what) && e.blockedURI === "data")) return;
    policyHit = what || "policy";
  };
  // reuse: the client that was kept mounted after "unsupported" has become ready (lateStart): the same frame goes live
  const start = (reuse) => {
    cancelLoop();
    if (!reuse) policyHit = null;
    if (watch) clearTimeout(watch);
    if (lateWatch) clearInterval(lateWatch);
    lateWatch = null;
    if (frame && !reuse) { frame.remove(); frame = null; }
    root.classList.remove("vx-late");
    showDetail("", "");
    if (againBtn) againBtn.hidden = true;
    clocks = [];
    video.pause();
    // the picture the video view shows stays on the canvas while the client loads (vx-held); else the poster shows
    root.classList.toggle("vx-held", mode === "video" && root.classList.contains("vx-drawn") && vcan.width > 0);
    noPoster();
    mode = "starting";
    root.classList.remove("vx-drawn");
    stage.style.aspectRatio = "";
    root.classList.add("is-live");      // the stage takes its live size before the field of view is set
    video.controls = false;
    waitFor(STATUS.live);
    note.hidden = true;
    sw.hidden = true;
    labelFb.hidden = true;
    labelLive.hidden = false;
    setSub(true);
    reset.hidden = false;
    opacityBtn.hidden = false;          // in place from the start (enabled once the client is driven): nothing moves later
    opacityBtn.disabled = true;
    if (modeBtn) { modeBtn.hidden = false; modeBtn.textContent = "Pre-rendered video"; }
    const rec = `viewer/recordings/${key}.viser`;
    if (!reuse) {
      frame = document.createElement("iframe");
      frame.className = "vx-frame";
      frame.title = "Interactive 3D viewer: HandReel's posed object mesh and the estimated hand";
      frame.setAttribute("allow", "fullscreen");
      frame.src = `viewer/client/?playbackPath=../recordings/${key}.viser&hideViserLogo&initialCameraFov=${fovFor().toFixed(4)}` +
        `&initialCameraLookAt=${root.dataset.look}`;   // along the optical axis: the image plane faces the camera
      stage.insertBefore(frame, stage.firstChild);
      startedAt = Date.now();
    }
    const f = frame;
    const t0 = Date.now();
    let last = -1;
    const stall = activeClock();        // visible, painted time without progress of the recording download
    const mismatch = activeClock();     // visible, painted time during which a client internal has been missing
    clocks.push(stall, mismatch);
    let arrivedAt = 0;
    const check = () => {
      if (f !== frame) return;
      let d = null;
      try { d = f.contentDocument; } catch (e) { d = null; }
      const now = Date.now();
      if (f.contentWindow && !f.contentWindow.__vxQuiet) {
        f.contentWindow.__vxQuiet = true;
        // the client's unhandled promise rejections stay out of the console (for example its unused splat-sorting
        // module, whose WebAssembly glue calls Function() and is refused by a policy without 'unsafe-eval')
        f.contentWindow.addEventListener("unhandledrejection", (e) => e.preventDefault());
        f.contentWindow.addEventListener("securitypolicyviolation", onViolation);
      }
      if (policyHit) { fallBack("blocked", `content policy: ${policyHit}`); return; }       // the policy refused something the client needs
      const arrived = d && d.readyState === "complete" && d.getElementById("root");
      if (!arrived) {                                        // the client's own page is still arriving
        // while the client's page arrives the check runs often, so that the two listeners above are on the new
        // window before the client's own code starts
        if (now - t0 < PAGE_CAP_MS) { watch = setTimeout(check, now - t0 < 5000 ? 20 : 250); return; }
        fallBack("stalled", "the client's page did not arrive");
        return;
      }
      if (!arrivedAt) arrivedAt = now;
      // the recording request answered with an error status (Resource Timing in the client's frame): give up now
      const recUrl = new URL(rec, document.baseURI).href;
      const ent = (f.contentWindow.performance.getEntriesByName(recUrl) || [])[0];
      if (ent && ent.responseStatus >= 400) { fallBack("failed", `recording answered ${ent.responseStatus}`); return; }
      const r = liveSetup(f.contentWindow, d);
      if (r === "ok") { watch = null; requestAnimationFrame(liveTick); return; }
      if (r === "noinput") { fallBack("failed", "the input frame list did not load"); return; }
      if (r.startsWith("mismatch")) {        // parts can still be initialising: refuse only if it persists
        // ... for MISMATCH_MS of visible, painted time. The client's camera, scene and reset function exist only
        // after its canvas has been measured and drawn once, which a hidden or throttled tab never does.
        if (mismatch(f.contentWindow) > MISMATCH_MS) {
          root.dataset.clientMismatch = r;
          fallBack("unsupported", `client not ready: ${r.replace(/^mismatch: /, "")} missing`, true);   // the client stays mounted
          lateStart(f);
          return;
        }
        watch = setTimeout(check, 250);
        return;
      }
      mismatch.reset();
      // The client's page is here but its own script has not run (its loader publishes __viserZstdWasm first): a
      // content policy refused the inline script or WebAssembly, or the script failed. A refusal this page saw has
      // already ended in "blocked" above; one reported before this page could listen is not known to be the cause.
      if (!f.contentWindow.__viserZstdWasm && now - arrivedAt > BOOT_MS) {
        fetch(f.src, { method: "HEAD" })
          .then((res) => {
            // A policy header on the client's page does not show that the policy stopped the script: the note
            // names the policy only when this page saw a refusal (policyHit, above). Here it is the general note;
            // the header's presence goes into the detail line as a fact.
            const has = !!res.headers.get("Content-Security-Policy");
            if (f === frame) fallBack("failed", `the client's script did not run${has ? " (its page is served with a content-policy header)" : ""}`);
          })
          .catch(() => { if (f === frame) fallBack("failed", "the client's script did not run"); });
        return;
      }
      // the recording is in (the playback bar is there) and only the page's own input frames are missing: that is
      // not the client stalling; a failed input list ends in "noinput" above
      if (!input && d.querySelector(".mantine-Slider-root")) { stall.reset(); watch = setTimeout(check, 250); return; }
      const bar = d.querySelector("[role=progressbar]");
      const got = bar ? parseFloat(bar.getAttribute("aria-valuenow") || "-1") : -1;
      if (got > last) { last = got; stall.reset(); }
      if (stall(f.contentWindow) < STALL_MS) { watch = setTimeout(check, 250); return; }
      fallBack("stalled", `recording download stopped at ${last < 0 ? "its start" : last + "%"}`);
    };
    // a client page that is not the client (an error page) gives up at once
    f.addEventListener("load", () => {
      if (f !== frame) return;
      let ok = false;
      try { ok = !!f.contentDocument.getElementById("root"); } catch (e) { ok = false; }
      if (!ok) fallBack("failed", "the client's page is not the viewer");
    });
    f.addEventListener("error", () => { if (f === frame) fallBack("failed", "the client's page failed to load"); });
    watch = setTimeout(check, 20);
  };
  // "unsupported" is not final while the client is alive: it stays mounted behind the video (same size, unseen), its
  // internals are checked once a second for LATE_MS, and when all are there the same client goes live, with the
  // frame on screen and the play state kept as in a view switch. Seconds in a hidden tab are not counted.
  const lateStart = (f) => {
    let left = Math.round(LATE_MS / 1000);
    lateWatch = setInterval(() => {
      if (f !== frame || mode !== "video") { clearInterval(lateWatch); lateWatch = null; return; }
      if (document.visibilityState !== "visible") return;
      let ready = false;
      try {
        const d = f.contentDocument;
        const ctx = d && d.querySelector(".mantine-Slider-root") ? findContext(d) : null;
        ready = !!ctx && input && checkClient(d, ctx) === null && !policyHit;
      } catch (e) { ready = false; }
      if (ready) { handOver(); start(true); return; }
      left -= 1;
      if (left <= 0) {                     // given up for good: the client goes, the note and the retry button stay
        clearInterval(lateWatch);
        lateWatch = null;
        f.remove();
        if (frame === f) frame = null;
        root.classList.remove("vx-late");
      }
    }, 1000);
  };
  // Touch screens: a transparent layer over the client takes touches (touch-action pan-y), so a vertical swipe
  // scrolls the page (a touch inside the iframe would not reach the page), and a one-finger horizontal swipe orbits
  // the camera about the vertical axis through the client's own camera controls (same limits).
  if (window.matchMedia("(pointer: coarse)").matches) root.classList.add("touch");   // primary pointer is touch
  const touchLayer = document.createElement("div");
  touchLayer.className = "vx-touch";
  touchLayer.setAttribute("aria-hidden", "true");
  stage.appendChild(touchLayer);
  const touches = new Map();
  touchLayer.addEventListener("pointerdown", (e) => { if (e.pointerType === "touch") touches.set(e.pointerId, [e.clientX, e.clientY]); });
  const spread = () => { const [a, b] = [...touches.values()]; return Math.hypot(a[0] - b[0], a[1] - b[1]); };
  touchLayer.addEventListener("pointermove", (e) => {
    if (!touches.has(e.pointerId) || mode !== "live" || !live) return;
    const c = live.ctx.mutable.current.cameraControl;
    if (touches.size === 2) {                              // two fingers: pinch zooms (clamped like Ctrl + wheel), no orbit
      const before = spread();
      touches.set(e.pointerId, [e.clientX, e.clientY]);
      const after = spread();
      if (before > 0 && after > 0) {
        c.dollyTo(clampD(c, live.limits, c.distance * Math.max(0.6, Math.min(1.6, before / after))), false);
        live.ctx.mutable.current.requestRender();
      }
      return;
    }
    const prev = touches.get(e.pointerId)[0];
    touches.set(e.pointerId, [e.clientX, e.clientY]);
    if (touches.size !== 1) return;                        // three or more fingers: nothing
    c.rotate((-(e.clientX - prev) / Math.max(1, stage.clientWidth)) * ORBIT.az * 1.5, 0, false);
    live.ctx.mutable.current.requestRender();
  });
  ["pointerup", "pointercancel", "pointerleave"].forEach((ev) => touchLayer.addEventListener(ev, (e) => touches.delete(e.pointerId)));

  reset.addEventListener("click", () => {
    if (mode === "live" && live) live.ctx.mutable.current.resetCameraPose(true);
  });
  opacityBtn.addEventListener("click", () => {
    seeThrough = !seeThrough;
    opacityBtn.setAttribute("aria-pressed", seeThrough ? "true" : "false");
  });
  if (againBtn) {
    againBtn.addEventListener("click", () => {   // after an automatic fallback: a fresh client, same frame and play state
      if (offline || liveOff || webgl === "none") return;
      started = true;
      handOver();
      start(); playBtn.focus();
    });
  }
  tryBtn.addEventListener("click", () => {   // keeps the shown frame, like the Video / 3D view switch
    if (offline || liveOff) return;
    started = true;
    handOver();
    start(); playBtn.focus();
  });
  // below 600 px the input frame is a small inset in the stage (compare without scrolling); the same element,
  // so its frame stays in sync; moved back into the grid on wider screens
  const vin = root.querySelector(".vx-in");
  const vgrid = root.querySelector(".vx-grid");
  const phone = window.matchMedia("(max-width: 599px)");
  const placeInput = () => {
    if (!vin || !vgrid) return;
    // above the stage, not over it, so in the pre-rendered video it sits right next to the camera view
    if (phone.matches) { if (vin.nextElementSibling !== stage) stage.before(vin); root.classList.add("in-inset"); }
    else if (vin.parentElement !== vgrid) { vgrid.insertBefore(vin, vgrid.firstChild); root.classList.remove("in-inset"); }
  };
  placeInput();
  phone.addEventListener("change", placeInput);
  // on phones (<= 760 px) Play, Replay and the slider sit above the 3D panel (CSS order); the
  // readout and the secondary toggles move to a second row under the stage, so the stage still starts on the first
  // screen. They move back on wider screens.
  const phoneCtl = window.matchMedia("(max-width: 760px)");
  const ctl = root.querySelector(".vx-ctl");
  const ctl2 = document.createElement("div");
  ctl2.className = "vx-ctl2";
  const movable = [".vx-excerpt", ".vx-readout", ".vx-opacity", ".vx-reset", ".vx-mode", ".vx-intoggle"].map((q) => root.querySelector(q)).filter(Boolean);
  const excerptHome = root.querySelector(".vx-clip");   // the excerpt line returns to the bar on wide screens
  const placeControls = () => {
    if (!ctl || !vgrid) return;
    if (phoneCtl.matches) {
      if (!ctl2.isConnected) vgrid.after(ctl2);
      for (const el of movable) if (el.parentElement !== ctl2) ctl2.appendChild(el);
    } else {
      for (const el of movable) {
        if (el.classList.contains("vx-excerpt")) { if (el.parentElement !== excerptHome) excerptHome.appendChild(el); }
        else if (el.parentElement !== ctl) ctl.appendChild(el);
      }
      if (ctl2.isConnected) ctl2.remove();
    }
  };
  placeControls();
  phoneCtl.addEventListener("change", placeControls);
  const inToggle = root.querySelector(".vx-intoggle");   // phones: the input video is shown on request
  if (inToggle) {
    inToggle.addEventListener("click", () => {
      const on = !root.classList.contains("show-in");
      root.classList.toggle("show-in", on);
      inToggle.setAttribute("aria-pressed", on ? "true" : "false");
      inToggle.textContent = on ? "Hide input" : "Input video";
    });
  }
  if (modeBtn) {
    let leaving = null;
    modeBtn.addEventListener("click", () => {
      started = true;
      if (mode === "live" && live) {
        // 3D -> video: the frame on screen and the play state are kept. The client's next render is copied to the
        // canvas (with the frame it shows), so the stage holds that picture until the video's own first frame.
        if (leaving) return;
        handOver();
        const lv = live;
        const go = () => {
          if (leaving === null) return;
          clearTimeout(leaving);
          leaving = null;
          if (mode === "live" && live === lv) { fallBack("chosen"); modeBtn.focus(); }
        };
        leaving = setTimeout(go, 400);            // no render came (a hidden tab): switch without the held picture
        lv.snap = (canvas) => {
          try {
            const refs = lv.ctx.mutable.current.nodeRefFromName;
            const i = input.frames.findIndex((f) => { const o = refs[`/f${String(f.frame).padStart(3, "0")}`]; return o && o.visible; });
            vcan.width = canvas.width;
            vcan.height = canvas.height;
            vcan.getContext("2d").drawImage(canvas, 0, 0);
            onCanvas = -1;
            if (i >= 0) { carry = i; showFrame(i); root.classList.add("vx-held"); }
          } catch (e) { root.classList.remove("vx-held"); }
          setTimeout(go, 0);
        };
        lv.ctx.mutable.current.requestRender();
      } else if (mode !== "live" && !offline && !liveOff && webgl !== "none") { handOver(); start(); playBtn.focus(); }
    });
  }

  const embedded = root.dataset.input ? JSON.parse(root.dataset.input) : null;   // offline build only
  // from file:// nothing is fetched: the embedded list, or (without one) the plain video
  (embedded ? Promise.resolve(embedded) : offline ? Promise.reject(new Error("file://"))
    : fetch(`viewer/input/${key}/index.json`).then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status))))))
    .then((ix) => {
      const base = `viewer/input/${key}/`;
      // loaded, not decode(): in a tab that is not being painted decode() can stay pending for as long as the tab is
      // hidden, and the viewer would wait for its input frames all that time (drawImage decodes when it draws)
      const imgs = ix.frames.map((x) => { const e = new Image(); e.src = base + x.src; return e; });
      const loaded = (e) => (e.complete && e.naturalWidth ? Promise.resolve() : new Promise((ok, no) => { e.onload = ok; e.onerror = no; }));
      return Promise.all(imgs.map(loaded)).then(() => {
        input = { frames: ix.frames, imgs, dur: ix.duration || ix.frames[ix.frames.length - 1].t };
        seek.max = String(ix.frames.length - 1);
        root.classList.add("vx-inp");      // the input panel is the page's canvas from now on, in every state
        const at = shown;                  // a frame kept across a view switch made before the frames arrived
        shown = -1;
        showFrame(at > 0 && at < ix.frames.length ? at : 0);
        if (mode === "video") drawnMode();
        if (visible) autoStart();
      });
    })
    .catch(() => {
      inputFailed = true;
      if (mode === "video") plainVideo();      // live: check() sees inputFailed and falls back to the plain video
    });

  // the frame marks of the pre-rendered videos (a small file); without them the video mode runs on the video's clock
  if (!marksAll && !offline) {
    fetch(`viewer/fallback/${key}.marks.json`).then((r) => (r.ok ? r.json() : null))
      .then((m) => { marksAll = m; if (m && mode === "video" && root.classList.contains("vx-drawn")) { useMarks(); paintVideo(true); } })
      .catch(() => {});
  }

  if (liveOff) fallBack("off");
  else if (offline) fallBack("offline");
  else if (webgl !== "none" && (forced || webgl === "hardware")) start();   // without WebGL the client is never started
  else fallBack(webgl === "software" ? "software" : "nowebgl");
  if (webgl === "none" || offline || liveOff) { tryBtn.hidden = true; sw.hidden = true; }
};
  // The script is loaded in the head without defer and starts as soon as the viewer's markup is parsed: it looks
  // for it in every animation frame, whose callbacks run before the browser paints that frame, so the controls, the
  // view the page chooses and the stage's size are in place at the first paint and nothing moves afterwards. If the
  // markup is already parsed it starts at once; with no viewer in the page, or in a tab that paints nothing, when the
  // document is parsed.
  const parsed = () => { const h = document.querySelector(".hero-viewer"); return !!h && !!h.nextElementSibling; };
  let booted = false;
  const go = () => { if (!booted) { booted = true; boot(); } };
  if (document.readyState !== "loading" || parsed()) go();
  else {
    const look = () => { if (booted) return; if (parsed()) go(); else requestAnimationFrame(look); };
    requestAnimationFrame(look);
    document.addEventListener("DOMContentLoaded", go);
  }
})();
