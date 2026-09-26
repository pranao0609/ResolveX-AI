"""
expert_resolvers.py — Contains the category-wise pool of expert resolvers with specializations.
"""

from typing import Dict, List, Any
import itertools
import re

EXPERT_RESOLVERS_POOL: Dict[str, List[Dict[str, Any]]] = {
    "software": [
        {
            "id": "res_sw_01",
            "name": "George Hacker",
            "category": "software",
            "specialization": ["database", "sql", "query", "data", "postgres"],
        },
        {
            "id": "res_sw_02",
            "name": "Hannah Code",
            "category": "software",
            "specialization": ["frontend", "ui", "react", "button"],
        },
        {
            "id": "res_sw_03",
            "name": "Ian System",
            "category": "software",
            "specialization": ["backend", "api", "endpoint", "server", "500", "404"],
        },
        {
            "id": "res_sw_04",
            "name": "Julia Scripts",
            "category": "software",
            "specialization": ["mobile", "app", "ios", "android", "crash"],
        },
        {
            "id": "res_sw_05",
            "name": "Kevin Nodes",
            "category": "software",
            "specialization": ["performance", "slow", "timeout", "latency", "lag"],
        },
        {
            "id": "res_sw_06",
            "name": "Alice Patch",
            "category": "software",
            "specialization": ["integration", "webhook", "sync", "bug", "glitch"],
        },
        {
            "id": "res_sw_07",
            "name": "David Glitch",
            "category": "software",
            "specialization": ["browser", "chrome", "safari", "firefox", "edge"],
        },
    ],
    "hardware": [
        {
            "id": "res_hw_01",
            "name": "Steve Planner",
            "category": "hardware",
            "specialization": ["mouse", "keyboard", "peripheral"],
        },
        {
            "id": "res_hw_02",
            "name": "Tina Roadmap",
            "category": "hardware",
            "specialization": ["disk", "storage", "drive", "hdd", "ssd"],
        },
        {
            "id": "res_hw_03",
            "name": "Ursula Backlog",
            "category": "hardware",
            "specialization": ["screen", "monitor", "display", "flicker"],
        },
        {
            "id": "res_hw_04",
            "name": "Victor Sprint",
            "category": "hardware",
            "specialization": ["battery", "power", "charger", "psu"],
        },
        {
            "id": "res_hw_05",
            "name": "Wendy Vision",
            "category": "hardware",
            "specialization": ["ram", "memory", "cpu", "overheat"],
        },
        {
            "id": "res_hw_06",
            "name": "Xavier Scope",
            "category": "hardware",
            "specialization": ["usb", "port", "jack", "audio", "speaker"],
        },
    ],
    "network": [
        {
            "id": "res_net_01",
            "name": "Yara Tracker",
            "category": "network",
            "specialization": ["wifi", "wireless", "signal"],
        },
        {
            "id": "res_net_02",
            "name": "Zane Fixer",
            "category": "network",
            "specialization": ["router", "modem", "switch", "gateway"],
        },
        {
            "id": "res_net_03",
            "name": "Frank Support",
            "category": "network",
            "specialization": ["vpn", "proxy", "tunnel"],
        },
        {
            "id": "res_net_04",
            "name": "Bob Debug",
            "category": "network",
            "specialization": ["dns", "ip", "address", "conflict"],
        },
        {
            "id": "res_net_05",
            "name": "Charlie Error",
            "category": "network",
            "specialization": ["lan", "cable", "ethernet", "drop"],
        },
        {
            "id": "res_net_06",
            "name": "Grace Guide",
            "category": "network",
            "specialization": ["firewall", "port", "blocking", "bandwidth"],
        },
    ],
    "access_permission": [
        {
            "id": "res_acc_01",
            "name": "Mike Profile",
            "category": "access_permission",
            "specialization": ["password", "reset", "forgot"],
        },
        {
            "id": "res_acc_02",
            "name": "Nina Login",
            "category": "access_permission",
            "specialization": ["login", "signin", "lockout", "locked"],
        },
        {
            "id": "res_acc_03",
            "name": "Oscar Auth",
            "category": "access_permission",
            "specialization": ["2fa", "mfa", "otp", "token"],
        },
        {
            "id": "res_acc_04",
            "name": "Penny Pass",
            "category": "access_permission",
            "specialization": ["role", "permission", "rights"],
        },
        {
            "id": "res_acc_05",
            "name": "Quinn Access",
            "category": "access_permission",
            "specialization": ["session", "timeout", "cookie"],
        },
        {
            "id": "res_acc_06",
            "name": "Rachel Identity",
            "category": "access_permission",
            "specialization": ["account", "disabled", "expired", "unlock"],
        },
    ],
    "security": [
        {
            "id": "res_sec_01",
            "name": "Arthur Billington",
            "category": "security",
            "specialization": ["breach", "hack", "leak", "vulnerability"],
        },
        {
            "id": "res_sec_02",
            "name": "Beatrice Cash",
            "category": "security",
            "specialization": ["malware", "virus", "scan", "script"],
        },
        {
            "id": "res_sec_03",
            "name": "Cecil Coin",
            "category": "security",
            "specialization": ["phishing", "spam", "suspicious", "attempt"],
        },
        {
            "id": "res_sec_04",
            "name": "Diana Ledger",
            "category": "security",
            "specialization": ["certificate", "ssl", "tls", "encryption"],
        },
        {
            "id": "res_sec_05",
            "name": "Edward Funds",
            "category": "security",
            "specialization": ["brute", "force", "ip", "block"],
        },
        {
            "id": "res_sec_06",
            "name": "Fiona Wealth",
            "category": "security",
            "specialization": ["patch", "update", "audit", "compliance"],
        },
    ],
    "other": [
        {
            "id": "res_oth_01",
            "name": "Eve Helper",
            "category": "other",
            "specialization": ["billing", "invoice", "refund"],
        },
        {
            "id": "res_oth_02",
            "name": "Henry Ticket",
            "category": "other",
            "specialization": ["training", "guide", "tutorial"],
        },
        {
            "id": "res_oth_03",
            "name": "Ivy Assist",
            "category": "other",
            "specialization": ["feedback", "complaint", "feature"],
        },
        {
            "id": "res_oth_04",
            "name": "Jack Resolve",
            "category": "other",
            "specialization": ["upgrade", "subscription", "plan"],
        },
        {
            "id": "res_oth_05",
            "name": "Karen Care",
            "category": "other",
            "specialization": ["documentation", "information", "how"],
        },
        {
            "id": "res_oth_06",
            "name": "Leo Empathy",
            "category": "other",
            "specialization": ["demo", "escalate", "status"],
        },
        {
            "id": "res_oth_07",
            "name": "Mia Service",
            "category": "other",
            "specialization": ["general", "misc", "unknown"],
        },
        {
            "id": "res_oth_08",
            "name": "Nathan Omni",
            "category": "other",
            "specialization": ["account", "deletion", "data", "export"],
        },
    ],
}

_ROUND_ROBIN_ITERATORS = {
    category: itertools.cycle(resolvers)
    for category, resolvers in EXPERT_RESOLVERS_POOL.items()
}


def get_best_expert_resolver(category: str, ticket_text: str) -> Dict[str, Any]:
    """
    Returns the best matching expert resolver for the specified category based on
    keyword matching against their specialization. If no keyword matches,
    it falls back to round-robin assignment within that category.
    """
    if category not in EXPERT_RESOLVERS_POOL:
        category = "other"

    resolvers = EXPERT_RESOLVERS_POOL[category]
    text_lower = ticket_text.lower()

    # 1. Try to find a resolver by matching specialization keywords
    best_resolver = None
    max_matches = 0

    for resolver in resolvers:
        matches = sum(
            1 for keyword in resolver["specialization"] if keyword in text_lower
        )
        if matches > max_matches:
            max_matches = matches
            best_resolver = resolver

    # 2. If we found a specialization match, use it. Otherwise fallback to round-robin
    if best_resolver and max_matches > 0:
        return best_resolver

    return next(_ROUND_ROBIN_ITERATORS[category])
