// Byte-level BPE for the released 1,024-token Fly LLM vocabulary.
// The vocabulary carries only 765 merges, so a direct implementation is smaller and
// clearer than pulling in a tokenizer library, and it keeps the demo dependency-free.

function byteToUnicode() {
  const printable = [];
  for (let i = 33; i <= 126; i += 1) printable.push(i);
  for (let i = 161; i <= 172; i += 1) printable.push(i);
  for (let i = 174; i <= 255; i += 1) printable.push(i);
  const bytes = printable.slice();
  const codes = printable.slice();
  const taken = new Set(printable);
  let extra = 0;
  for (let b = 0; b < 256; b += 1) {
    if (!taken.has(b)) {
      bytes.push(b);
      codes.push(256 + extra);
      extra += 1;
    }
  }
  const forward = new Map();
  const backward = new Map();
  bytes.forEach((b, i) => {
    const ch = String.fromCodePoint(codes[i]);
    forward.set(b, ch);
    backward.set(ch, b);
  });
  return { forward, backward };
}

// The GPT-2 pre-tokenizer pattern; the released tokenizer declares a ByteLevel
// pre-tokenizer with use_regex true and add_prefix_space false.
const PATTERN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export class FlyTokenizer {
  constructor(spec) {
    const { forward, backward } = byteToUnicode();
    this.byteToChar = forward;
    this.charToByte = backward;
    this.vocab = spec.model.vocab;
    this.ranks = new Map();
    spec.model.merges.forEach((pair, index) => {
      const [a, b] = Array.isArray(pair) ? pair : pair.split(' ');
      this.ranks.set(`${a} ${b}`, index);
    });
    this.idToToken = [];
    for (const [token, id] of Object.entries(this.vocab)) this.idToToken[id] = token;
    for (const added of spec.added_tokens || []) this.idToToken[added.id] = added.content;
    this.special = new Set((spec.added_tokens || []).map((added) => added.id));
    this.padId = 0;
    this.bosId = 1;
    this.eosId = 2;
    this.utf8 = new TextEncoder();
    this.utf8Decoder = new TextDecoder('utf-8', { fatal: false });
  }

  static async load(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`tokenizer request failed: ${response.status}`);
    return new FlyTokenizer(await response.json());
  }

  mergeWord(word) {
    let symbols = Array.from(word);
    if (symbols.length < 2) return symbols;
    for (;;) {
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i < symbols.length - 1; i += 1) {
        const rank = this.ranks.get(`${symbols[i]} ${symbols[i + 1]}`);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          best = i;
        }
      }
      if (best < 0) return symbols;
      symbols = symbols
        .slice(0, best)
        .concat(symbols[best] + symbols[best + 1], symbols.slice(best + 2));
      if (symbols.length === 1) return symbols;
    }
  }

  encode(text) {
    const ids = [];
    for (const match of text.matchAll(PATTERN)) {
      let mapped = '';
      for (const byte of this.utf8.encode(match[0])) mapped += this.byteToChar.get(byte);
      for (const piece of this.mergeWord(mapped)) {
        const id = this.vocab[piece];
        // Every single byte is in the vocabulary, so a completed merge cannot leave an
        // unknown piece behind. Fail loudly rather than emit undefined if that changes.
        if (id === undefined) throw new Error(`token not in vocabulary: ${piece}`);
        ids.push(id);
      }
    }
    return ids;
  }

  // Decoding one id at a time can split a multi-byte character, so callers decode the
  // whole continuation each step and diff against what they already displayed.
  decode(ids, skipSpecial = true) {
    const bytes = [];
    for (const id of ids) {
      if (skipSpecial && this.special.has(id)) continue;
      const token = this.idToToken[id];
      if (token === undefined) continue;
      for (const ch of token) {
        const byte = this.charToByte.get(ch);
        if (byte !== undefined) bytes.push(byte);
      }
    }
    return this.utf8Decoder.decode(new Uint8Array(bytes));
  }
}
