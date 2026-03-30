const axios = require('axios');

async function test(prompt) {
  const res = await axios.post('http://localhost:8000/generate', { prompt });
  console.log(prompt, '->', res.data.source, '| saved:', res.data.saved);
}

async function main() {
  await test('What is the capital of France?');
  await test('Capital city of France?');
  await test('How do I boil an egg?');
}

main();
