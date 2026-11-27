"""Tokenizer probe battery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

BASE = "The quick brown fox.\n"
TINY = 1


@dataclass
class Probe:
    id: str
    channel: str
    body: Dict
    raw: bool = False
    stream: bool = False
    note: str = ""


def _msg_probe(pid: str, text: str, note: str = "") -> Probe:
    return Probe(pid, "tokenizer",
                 {"messages": [{"role": "user", "content": BASE + text}],
                  "max_tokens": TINY, "temperature": 0.0},
                 note=note)


TOKENIZER_PROBES: Dict[str, str] = {
    "en_pangram": "Pack my box with five dozen liquor jugs, quickly and quietly, before dawn breaks over the harbor.",
    "en_long": "Assessment uniformity requires that comparable properties within the same class receive comparable valuations, a principle routinely tested in certiorari proceedings.",
    "zh_simp": "人工智能正在深刻改变软件开发的方式，大模型的推理能力决定了智能体的上限。",
    "zh_classical": "落霞与孤鹜齐飞，秋水共长天一色。",
    "zh_mixed": "这个 API 的 rate limit 是每天 1000 次请求，超过就会返回 429 错误。",
    "ja": "東京の秋は紅葉が美しく、京都では寺院の庭園が観光客で賑わいます。",
    "ko": "서울의 가을은 단풍이 아름답고 남산 타워에서 야경을 볼 수 있습니다.",
    "ar": "الذكاء الاصطناعي يغير طريقة تطوير البرمجيات بشكل جذري وعميق.",
    "ru": "Искусственный интеллект глубоко меняет способы разработки программного обеспечения.",
    "hi": "कृत्रिम बुद्धिमत्ता सॉफ्टवेयर विकास के तरीके को गहराई से बदल रही है।",
    "th": "ปัญญาประดิษฐ์กำลังเปลี่ยนแปลงวิธีการพัฒนาซอฟต์แวร์อย่างลึกซึ้ง",
    "emoji": "👨\u200d👩\u200d👧\u200d👦🏳️\u200d🌈🇺🇸🇨🇳🫡🤌🏽",
    "code_py": "def parse(s: str) -> dict[str, int]:\n    return {k: int(v) for k, v in (p.split('=') for p in s.split('&'))}",
    "code_json": '{"nodes": [{"id": 1, "edges": [2, 3]}, {"id": 2, "edges": []}], "directed": true}',
    "code_sql": "SELECT p.pin, SUM(a.certified) FROM parcels p JOIN assessments a USING (pin) GROUP BY 1 HAVING SUM(a.certified) > 1e6;",
    "regex": r"^(?:[a-f0-9]{8}-[a-f0-9]{4}-[1-5][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12})$",
    "numbers": "3.14159265358979323846264338327950288419716939937510",
    "digits_grouped": "1,234,567,890.987654321 and 0x1F4A9 and 0b101010",
    "whitespace": "a    b\t\tc\n\n\nd     e",
    "markdown": "| PIN | Mailed | Certified |\n|-----|--------|-----------|\n| 12-27-300-066 | $98.5M | $34.4M |",
    "url": "https://openrouter.ai/api/v1/chat/completions?model=stealth%2Fox-alpha&limit=100",
    "base64": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dGVzdC1wYXlsb2FkLWZvci10b2tlbml6ZXI",
    "rare_cjk": "𠀀𠀁齉龘鱻麤靐飝",
    "box_drawing": "┌──┬──┐\n│ab│cd│\n├──┼──┤\n└──┴──┘",
    "es": "El veloz murciélago hindú comía feliz cardillo y kiwi mientras la cigüeña tocaba el saxofón.",
    "fr": "Portez ce vieux whisky au juge blond qui fume sur son île intérieure, près du quai.",
    "de": "Zwölf Boxkämpfer jagen Viktor quer über den großen Sylter Deich bei Nacht.",
    "pt": "Luís argüia à Júlia que «brações, fé, chá, óxido, pôr, zângão» eram palavras do português.",
    "it": "Quel vituperabile xenofobo zelante assaggia il whisky ed esclama: alleluja!",
    "tr": "Pijamalı hasta yağız şoföre çabucak güvendi ve ılık sütünü içti.",
    "vi": "Tôi muốn học cách xây dựng hệ thống phân tích dữ liệu lớn và tối ưu hiệu năng.",
    "id": "Kecerdasan buatan mengubah cara pengembangan perangkat lunak secara mendasar.",
    "pl": "Zażółć gęślą jaźń, bo pchnąć w tę łódź jeża lub ośm skrzyń fig.",
    "he": "בינה מלאכותית משנה באופן עמוק את הדרך שבה מפתחים תוכנה.",
    "fa": "هوش مصنوعی روش توسعه نرم‌افزار را به‌طور بنیادی تغییر می‌دهد.",
    "el": "Η τεχνητή νοημοσύνη αλλάζει ριζικά τον τρόπο ανάπτυξης λογισμικού.",
    "bn": "কৃত্রিম বুদ্ধিমত্তা সফটওয়্যার উন্নয়নের পদ্ধতি মৌলিকভাবে বদলে দিচ্ছে।",
    "ta": "செயற்கை நுண்ணறிவு மென்பொருள் உருவாக்கத்தை அடிப்படையில் மாற்றுகிறது.",
    "uk": "Штучний інтелект докорінно змінює підходи до розробки програмного забезпечення.",
    "sw": "Akili bandia inabadilisha kabisa jinsi programu zinavyoundwa na kutengenezwa.",
    "ha": "Hankali na wucin gadi yana canza yadda ake kirkirar manhajo sosai.",
    "am": "ሰው ሰራሽ አስተውሎት የሶፍትዌር ልማት ዘዴን በመሠረቱ ይለውጣል።",
    "code_rust": "fn main() -> Result<(), Box<dyn std::error::Error>> { let v: Vec<u32> = (0..10).map(|x| x * x).collect(); println!(\"{:?}\", v); Ok(()) }",
    "code_go": "func parse(s string) (map[string]int, error) {\n\tout := make(map[string]int)\n\tfor _, p := range strings.Split(s, \"&\") { out[p] = 1 }\n\treturn out, nil\n}",
    "code_ts": "const f = async <T,>(x: readonly T[]): Promise<T | undefined> => x.at(-1);",
    "code_java": "public final class A<T extends Comparable<T>> implements Supplier<Optional<T>> { @Override public Optional<T> get() { return Optional.empty(); } }",
    "code_cpp": "template<typename T> constexpr auto f(T&& t) noexcept -> decltype(std::forward<T>(t).size()) { return t.size(); }",
    "code_shell": "for f in $(find . -name '*.py' -print0 | xargs -0 -n1 basename); do echo \"${f%.py}\"; done",
    "code_yaml": "services:\n  api:\n    image: ghcr.io/acme/api:1.2.3\n    environment:\n      - DEBUG=false\n    ports: [\"8080:8080\"]",
    "code_diff": "--- a/src/x.py\n+++ b/src/x.py\n@@ -1,4 +1,4 @@\n-def f(a, b):\n+def f(a: int, b: int) -> int:",
    "json_nested": "{\"a\":{\"b\":{\"c\":[1,2,3,{\"d\":null}]}},\"e\":[[],{},false]}",
    "csv": "id,name,score\n1,Ada Lovelace,99.5\n2,Grace Hopper,98.25\n3,Alan Turing,97.0",
    "uuid_hex": "550e8400-e29b-41d4-a716-446655440000 deadbeef 0xDEADBEEF 0b1010_1010",
    "scientific": "6.02214076e23 1.602176634e-19 -273.15 1_000_000 3.4e+38",
    "dates": "2026-11-19T12:40:00Z 19/11/2026 11/19/2026 2026年11月19日",
    "tabs_indent": "if x:\n\tif y:\n\t\treturn 1\n\treturn 2",
    "trailing_ws": "line one   \nline two\t\nline three  ",
    "nbsp": "alpha\u00a0beta\u2009gamma\u200bdelta\u200dfe",
    "combining": "e\u0301a\u0300o\u0308n\u0303u\u0308",
    "zwj_chain": "👩\u200d💻🧑🏽\u200d🚀🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "cjk_ext_b": "㐀㐁㐂㐃㐄㐅㐆㐇",
    "hangul_jamo": "ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉ",
    "math_symbols": "∑∏∫∮√∞≈≠≤≥∂∇∈∉⊂⊃∪∩∀∃",
    "greek": "αβγδεζηθικλμνξοπρστυφχψω",
    "ipa": "ˈɪntəˌnæʃəˈnælɪti ˌfəʊnəˈtɪk ˈælfəbɛt",
    "mixed_script": "東京 Tokyo 東京Токио 東京·Tokyo",
    "long_word": "pneumonoultramicroscopicsilicovolcanoconiosisandthenmoreletters",
    "repeated_char": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "punct_run": "!!!???...,,,;;;:::---___===+++***&&&|||",
    "parens": "(((((((((((((((((((((())))))))))))))))))))))",
    "html": "<div class=\"row\"><span id='x' data-y=\"1\">&amp; text</span></div>",
    "latex": "$\\int_0^\\infty e^{-x^2}\\,dx = \\frac{\\sqrt{\\pi}}{2}$",
    "ipv6_mac": "2001:0db8:85a3:0000:0000:8a2e:0370:7334 aa:bb:cc:dd:ee:ff",
    "base32": "MZXW6YTBOI======",
    "spaces_only": "     ",
    "newlines": "\n\n\n\n\n",
}


def tokenizer_probes() -> List[Probe]:
    probes = [Probe("tok:base", "tokenizer",
                    {"messages": [{"role": "user", "content": BASE}],
                     "max_tokens": TINY, "temperature": 0.0},
                    note="differential baseline")]
    probes += [_msg_probe(f"tok:{name}", text) for name, text in TOKENIZER_PROBES.items()]
    return probes
