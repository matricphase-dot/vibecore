const { generate } = require('./sdk');
const axios = require('axios');

async function main() {
  console.log('=== VIBECORE MVP FINAL TEST ===\n');

  console.log('--- Test 1: Simple prompt (Ollama) ---');
  const r1 = await generate('What is the capital of France?');
  console.log('Source:', r1.source, '| Complexity:', r1.complexity);
  console.log('Response:', r1.response);
  console.log('Saved: Rs.' + r1.saved, '| Total saved: Rs.' + r1.total_saved);

  console.log('\n--- Test 2: Same prompt (exact cache) ---');
  const r2 = await generate('What is the capital of France?');
  console.log('Source:', r2.source);
  console.log('Saved: Rs.' + r2.saved, '| Total saved: Rs.' + r2.total_saved);

  console.log('\n--- Test 3: Similar prompt (semantic cache) ---');
  const r3 = await generate('Capital city of France?');
  console.log('Source:', r3.source);
  console.log('Saved: Rs.' + r3.saved, '| Total saved: Rs.' + r3.total_saved);

  console.log('\n--- Test 4: Complex prompt (external API route) ---');
  const r4 = await generate('Analyze the difference between REST and GraphQL APIs in detail');
  console.log('Source:', r4.source, '| Complexity:', r4.complexity);
  console.log('Saved: Rs.' + r4.saved, '| Total saved: Rs.' + r4.total_saved);

  console.log('\n=== STATS ===');
  const stats = await axios.get('http://localhost:8000/stats');
  console.log(stats.data);
}

main();
