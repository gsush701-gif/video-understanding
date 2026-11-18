// Run inside an open YouTube watch page (browser tool's JavaScript runner).
// Opens the "Show transcript" panel, reads every line, and returns
//   "# <title> | <channel> | <seconds>s | captions: <tracks>\n[m:ss] text\n..."
// Throws NO_TRANSCRIPT when the video has no captions, LAYOUT_CHANGED when the
// panel opened but no lines could be read (YouTube changed its markup).
// Handles both the 2026 panel (transcript-segment-view-model) and the older one
// (ytd-transcript-segment-renderer). Direct caption URLs are not used: YouTube
// returns them empty without a page-issued token.
const sleep = ms => new Promise(r => setTimeout(r, ms));
document.querySelector('video')?.pause();

const grab = () => {
  let s = [...document.querySelectorAll('transcript-segment-view-model')];
  if (s.length) return s.map(e => [
    e.querySelector('[class*="SegmentViewModelTimestamp"]:not([class*="A11y"])')?.textContent.trim(),
    e.querySelector('span')?.textContent]);
  s = [...document.querySelectorAll('ytd-transcript-segment-renderer')];
  return s.map(e => [
    e.querySelector('.segment-timestamp')?.textContent.trim(),
    e.querySelector('.segment-text')?.textContent]);
};

await sleep(1000);
let rows = grab();
if (!rows.length) {
  document.querySelector('#description-inline-expander #expand, tp-yt-paper-button#expand')?.click();
  await sleep(700);
  const b = [...document.querySelectorAll('button')].find(x =>
    /show transcript/i.test((x.innerText || '') + (x.getAttribute('aria-label') || '')));
  if (!b) throw new Error('NO_TRANSCRIPT: this video has no captions/transcript');
  b.click();
  for (let i = 0; i < 40 && !rows.length; i++) { await sleep(500); rows = grab(); }
  if (!rows.length) throw new Error('LAYOUT_CHANGED: transcript panel opened but no lines found');
}
document.querySelector('video')?.pause();

const pr = document.querySelector('#movie_player')?.getPlayerResponse?.() || window.ytInitialPlayerResponse;
const d = pr?.videoDetails || {};
const tracks = (pr?.captions?.playerCaptionsTracklistRenderer?.captionTracks || [])
  .map(t => t.languageCode + (t.kind === 'asr' ? '(auto)' : '')).join(',');
const lines = rows.filter(r => r[1]).map(([t, x]) => `[${t}] ${x.replace(/\s+/g, ' ').trim()}`);
`# ${d.title} | ${d.author} | ${d.lengthSeconds}s | captions: ${tracks}\n` + lines.join('\n');
