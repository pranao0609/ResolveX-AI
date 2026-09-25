from ai.agents.query_analyzer import (
    analyze_query,
)


def test_normalizes_whitespace():
    result = analyze_query(
        "VPN    connection   fails    after login"
    )

    assert (
        result.normalized_query
        == "VPN connection fails after login"
    )


def test_short_query_uses_keyword_extraction():
    result = analyze_query(
        "VPN failure"
    )

    assert result.rewrite_needed is True
    assert result.rewrite_strategy == (
        "keyword_extraction"
    )
    assert result.retrieval_query


def test_error_query_is_rewritten():
    result = analyze_query(
        "My VPN connection fails after login"
    )

    assert result.rewrite_needed is True
    assert result.rewrite_strategy == (
        "technical_keyword_extraction"
    )

    assert "VPN" in result.retrieval_query
    assert "login" in result.retrieval_query


def test_normal_query_does_not_require_rewrite():
    result = analyze_query(
        "VPN client authentication session troubleshooting"
    )

    assert result.rewrite_needed is False
    assert result.rewrite_strategy == "none"

    assert (
        result.retrieval_query
        == result.normalized_query
    )


def test_ticket_id_noise_is_removed():
    result = analyze_query(
        "Ticket #12345 VPN connection failure"
    )

    assert "12345" not in result.retrieval_query


def test_conversational_noise_is_removed():
    result = analyze_query(
        "Please help me with my VPN connection failure"
    )

    assert result.rewrite_needed is True
    assert "VPN" in result.retrieval_query


def test_empty_query():
    result = analyze_query("")

    assert result.original_query == ""
    assert result.normalized_query == ""
    assert result.retrieval_query == ""
    assert result.rewrite_needed is False
    assert result.rewrite_strategy == "none"


def test_query_analysis_preserves_technical_terms():
    result = analyze_query(
        "Docker container cannot connect to PostgreSQL"
    )

    assert "Docker" in result.retrieval_query
    assert "PostgreSQL" in result.retrieval_query