/* The Video section (round B, 2026-11-23): the phone layout (9:16) on screens up to 760 px wide, the desktop layout
   (16:9) elsewhere. The <source media> order already picks the file where a browser honours it; this script sets src
   and poster explicitly, so every browser plays the file that matches the viewport, and sets the viewport's one poster (the markup
   carries none, so a phone does not also fetch the desktop poster). It never
   switches a video that the visitor has started. */
(() => {
  "use strict";
  const v = document.querySelector("#video video[data-src-m]");
  if (!v) return;
  const mq = window.matchMedia("(max-width: 760px)");
  const pick = () => {
    if (!v.paused || v.currentTime > 0) return;
    const phone = mq.matches;
    const src = phone ? v.dataset.srcM : v.dataset.src;
    const poster = phone ? v.dataset.posterM : v.dataset.poster;
    if (v.getAttribute("poster") !== poster) v.setAttribute("poster", poster);
    if (v.getAttribute("src") !== src) v.setAttribute("src", src);
  };
  pick();
  if (mq.addEventListener) mq.addEventListener("change", pick);
  else if (mq.addListener) mq.addListener(pick);
})();
