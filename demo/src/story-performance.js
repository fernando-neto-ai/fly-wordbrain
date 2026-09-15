/**
 * Illustrative story choreography. Reads the current token's text for pauses;
 * never changes generated text or reads neural activations/model parameters.
 *
 * All time inputs are supplied by the caller; no clocks, randomness or state.
 * `dt` is unscaled render time in seconds, clamped by the existing main loop.
 * Keep it unscaled so path and capped gait integration remain aligned.
 * `elapsedMs` and `frameDurationMs` use the displayed frame's wall-clock
 * timeline (already speed adjusted); punctuation pauses follow that pace.
 *
 * Manual Listen/Wander/Take flight modes bypass this function entirely.
 */
export function storyPerformance({
  playing,
  reducedMotion = false,
  frame = null,
  elapsedMs = 0,
  frameDurationMs = 360,
  dt = 0,
} = {}) {
  if (!playing || reducedMotion || !frame) {
    return { mode: 'idle', delta: 0 };
  }
  if (!Number.isFinite(dt) || dt < 0 || !Number.isFinite(elapsedMs)
      || !Number.isFinite(frameDurationMs) || frameDurationMs <= 0) {
    throw new RangeError('Story choreography requires finite nonnegative dt and a positive frame duration.');
  }
  if (frame.phase !== 'generation') {
    // Allow the existing rig to settle into its idle stance during prefill.
    return { mode: 'idle', delta: dt };
  }

  // Inspect only the prediction emitted by this frame. Testing the cumulative
  // prefix would repeatedly pause on an earlier period when no new text appears.
  const text = typeof frame.predicted_text === 'string' ? frame.predicted_text : '';
  const endsWithPunctuation = /[.!?,;:…](?:["'”’\)\]]*)\s*$/u.test(text);
  // At normal pace this is 180 ms; faster playback keeps some walking time in
  // the same frame instead of turning every punctuation frame into a full stop.
  const pauseMs = Math.min(180, frameDurationMs * 0.6);
  const pauseForPunctuation = endsWithPunctuation && Math.max(0, elapsedMs) < pauseMs;
  return { mode: pauseForPunctuation ? 'idle' : 'walk', delta: dt };
}
