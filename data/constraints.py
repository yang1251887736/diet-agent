"""饮食约束词表。

语料是英文，词表一律英文小写；匹配时对菜品文本 lower() 后做【词边界】匹配
（\\bword\\b），不是子串匹配——子串会把 eggplant 误判为 egg、goat 误判为 oat
（goat 在语料出现 7 次、oat 78 次，是真实风险）。

扩充方式：候选词先在本语料 1650 chunks 上 grep 验证频次，只保留真实出现过的词
（频次见注释）；个别语料未出现但同领域常见的词也保留，成本为零。
"""

MEAT = [  # 语料验证：meat 309 / chicken 197 / ham 101 / beef 80 / pork 76 / lamb 72 ...
    "meat", "chicken", "beef", "pork", "lamb", "veal", "turkey", "duck",
    "sausage", "ham", "bacon", "prosciutto", "meatball", "steak",
    "pancetta", "chorizo", "salami", "mortadella", "liver", "rabbit",
    "quail", "goose", "lard",
    # 法语/意大利语/中东菜系肉词（素食过滤曾漏放 Döner Kebab / Tête de veau / ossobuco）
    "kebab", "döner", "doner", "shawarma", "souvlaki", "gyro", "kofta",
    "ossobuco", "veau", "carne", "cotoletta", "schnitzel", "bolognese", "ragu", "ragù",
]
SEAFOOD = [  # anchovy 12 / crab 13 / octopus 11 / lobster 6 / bottarga 8 / roe 9 ...
    "fish", "seafood", "salmon", "tuna", "sardine", "anchovy", "anchovy paste",
    "shrimp", "prawn", "squid", "calamari", "octopus", "cuttlefish",
    "mussel", "clam", "cod", "scallop", "mackerel", "crab", "lobster",
    "roe", "caviar", "bottarga", "tarama",
]
DAIRY = [  # cheese 376 / butter 203 / cream 129 / parmesan 103 / milk 93 / yogurt 39 ...
    "milk", "cheese", "feta", "yogurt", "butter", "cream",
    "halloumi", "ricotta", "mozzarella", "parmesan", "labneh",
    "mascarpone", "pecorino", "brie", "manchego", "custard",
    "mayonnaise", "ghee", "whey", "casein", "goat cheese",
]
EGG = [  # egg 386 / omelet 31 / meringue 31 / custard 15 ...
    "egg", "omelet", "omelette", "meringue", "albumen", "mayonnaise",
    "hollandaise", "sabayon", "zabaione", "frittata", "quiche", "custard",
    "egg wash", "egg noodle", "beaten egg",
]
NUTS = [  # almond 46 / pine nut 39 / walnut 22 / pistachio 15 / peanut 6 / hazelnut 3 ...
    "almond", "walnut", "pine nut", "pistachio", "hazelnut", "cashew",
    "peanut", "pecan", "macadamia", "brazil nut", "marzipan", "nougat",
    "praline", "nut", "nuts",
]
GLUTEN = [  # bread 350 / pasta 274 / flour 221 / oat 78 / pita 70 / semolina 38 / couscous 27 ...
    "wheat", "barley", "bulgur", "couscous", "pasta", "bread", "flour",
    "semolina", "rye", "pita", "flatbread", "noodle", "pastry",
    "breadcrumb", "crouton", "oat", "oats", "oatmeal", "spelt",
    "sourdough", "dumpling",
]
ALCOHOL = [  # wine 127 / rum 86 / brandy 9 / vermouth 2 / ouzo 4 / marsala 7 / raki 2 ...
    "wine", "beer", "rum", "brandy", "liqueur", "ouzo", "alcohol",
    "marsala", "vermouth", "cider", "raki", "arak", "sherry", "grappa", "kirsch",
]

# 每种饮食偏好 → 禁止出现的词表并集
DIET_RULES = {
    "vegan": MEAT + SEAFOOD + DAIRY + EGG + ["honey"],
    "vegetarian": MEAT + SEAFOOD,
    "pescatarian": MEAT,
    "halal": ["pork", "bacon", "ham", "lard"] + ALCOHOL,
}

# ── 归一化：用户输的是自然语言，不是英文枚举 ──────────────────
import re as _re

PREF_ALIASES = {
    "vegetarian": "vegetarian", "素食": "vegetarian", "蛋奶素": "vegetarian",
    "vegan": "vegan", "纯素": "vegan", "全素": "vegan",
    "pescatarian": "pescatarian", "鱼素": "pescatarian",
    "halal": "halal", "清真": "halal",
}
# 中文/口语忌口 → 语料里的英文词（值可以是多个，命中任一即违规）
DISLIKE_ALIASES = {
    "辣": ["chili", "chilli", "spicy"], "辣椒": ["chili", "chilli"],
    "海鲜": ["seafood"], "贝类": ["shellfish", "mussel", "clam"],
    "坚果": ["nut", "nuts"], "花生": ["peanut"],
    "猪肉": ["pork"], "牛肉": ["beef"], "羊肉": ["lamb"],
    "乳制品": ["cheese", "milk", "butter", "cream"], "奶制品": ["cheese", "milk", "butter", "cream"],
    "蛋": ["egg"], "麸质": ["gluten", "wheat"],
}
_STRIP = _re.compile(r"^(不吃|不要|别吃|忌|不碰|no|not|without)\s*")


def normalize_prefs(prefs: list) -> list:
    """'素食'→'vegetarian' 等；无法识别的原样保留（宁可持续化不误删）。"""
    out = set()
    for p in prefs or []:
        p = p.strip().lower()
        if p:
            out.add(PREF_ALIASES.get(p, p))
    return sorted(out)


def normalize_dislikes(dislikes: list) -> list:
    """'不吃辣'→['chili','spicy'] 等：去掉'不吃'前缀 + 中文映射到英文词。"""
    out = set()
    for d in dislikes or []:
        d = _STRIP.sub("", d.strip().lower())
        if not d:
            continue
        for zh, en_list in DISLIKE_ALIASES.items():
            if zh in d:
                out.update(en_list)
                break
        else:
            out.add(d)   # 已是英文词的原样保留
    return sorted(out)
