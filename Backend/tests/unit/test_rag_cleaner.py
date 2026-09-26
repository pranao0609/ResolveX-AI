from datetime import datetime, timezone

from ai.rag.cleaner import DocumentCleaner
from ai.rag.models import CanonicalDocument


def make_document(
    title="VPN Troubleshooting",
    content="Restart the VPN client.",
):
    return CanonicalDocument(
        document_id="kb:1",
        source_id="article:1",
        title=title,
        content=content,
        category="network",
        source_type="article",
        created_at=datetime.now(timezone.utc),
        version="1.0",
    )


def test_cleaner_strips_title_whitespace():
    document = make_document(title="   VPN Troubleshooting   ")

    cleaned = DocumentCleaner.clean(document)

    assert cleaned.title == "VPN Troubleshooting"


def test_cleaner_removes_html_from_title():
    document = make_document(title="<h1>VPN Troubleshooting</h1>")

    cleaned = DocumentCleaner.clean(document)

    assert cleaned.title == "VPN Troubleshooting"


def test_cleaner_removes_html_from_content():
    document = make_document(content="<p>Restart the VPN client.</p>")

    cleaned = DocumentCleaner.clean(document)

    assert cleaned.content == "Restart the VPN client."


def test_cleaner_normalizes_line_endings():
    document = make_document(content="Step 1\r\nStep 2\r\nStep 3")

    cleaned = DocumentCleaner.clean(document)

    assert cleaned.content == "Step 1\nStep 2\nStep 3"


def test_cleaner_preserves_paragraph_boundaries():
    document = make_document(content="First paragraph.\n\nSecond paragraph.")

    cleaned = DocumentCleaner.clean(document)

    assert cleaned.content == "First paragraph.\n\nSecond paragraph."


def test_cleaner_collapses_excessive_blank_lines():
    document = make_document(content="First paragraph.\n\n\n\n\nSecond paragraph.")

    cleaned = DocumentCleaner.clean(document)

    assert cleaned.content == "First paragraph.\n\nSecond paragraph."


def test_cleaner_normalizes_multiple_spaces():
    document = make_document(content="Restart    the     VPN client.")

    cleaned = DocumentCleaner.clean(document)

    assert cleaned.content == "Restart the VPN client."


def test_cleaner_preserves_capitalization():
    document = make_document(content="Run PowerShell and execute Get-Service.")

    cleaned = DocumentCleaner.clean(document)

    assert cleaned.content == "Run PowerShell and execute Get-Service."


def test_cleaner_preserves_commands_and_paths():
    document = make_document(
        content="Run ipconfig /flushdns from C:\\Windows\\System32."
    )

    cleaned = DocumentCleaner.clean(document)

    assert "ipconfig /flushdns" in cleaned.content
    assert "C:\\Windows\\System32" in cleaned.content


def test_cleaner_does_not_modify_original_document():
    document = make_document(
        title="  VPN Troubleshooting  ",
        content="  Restart    VPN.  ",
    )

    cleaned = DocumentCleaner.clean(document)

    assert document.title == "  VPN Troubleshooting  "
    assert document.content == "  Restart    VPN.  "

    assert cleaned.title == "VPN Troubleshooting"
    assert cleaned.content == "Restart VPN."


def test_cleaner_handles_whitespace_only_content():
    document = make_document(content="   \n\n   ")

    cleaned = DocumentCleaner.clean(document)

    assert cleaned.content == ""


def test_cleaner_handles_whitespace_only_title():
    document = make_document(title="   ")

    cleaned = DocumentCleaner.clean(document)

    assert cleaned.title == ""
