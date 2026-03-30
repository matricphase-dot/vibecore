const axios = require('axios');

const BACKEND_URL = 'http://localhost:8000';

async function generate(prompt) {
  const start = Date.now();
  console.log('[VibeCore] Sending prompt: "' + prompt.slice(0, 60) + '..."');

  try {
    const response = await axios.post(BACKEND_URL + '/generate', { prompt });
    const ms = Date.now() - start;

    console.log('[VibeCore] Response in ' + ms + 'ms');
    console.log('[VibeCore] Source   : ' + response.data.source);
    console.log('[VibeCore] Cached   : ' + response.data.cached);
    console.log('[VibeCore] Saved    : Rs.' + response.data.saved);

    return response.data;
  } catch (err) {
    console.error('[VibeCore] Error:', err.message);
    throw err;
  }
}

module.exports = { generate };
