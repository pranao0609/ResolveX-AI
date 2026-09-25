"""
seed_data.py — Seed script to populate the database with sample tickets and KB entries.
               Also ingests KB entries into FAISS so RAG similarity scores are non-zero.
Run: python scripts/seed_data.py
"""

import sys
import os

# Ensure backend root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal, init_db
from app.models.ticket_model import Ticket
from app.models.kb_model import KnowledgeBaseEntry
from app.models.user_model import User



SAMPLE_TICKETS = [
    Ticket(
        title="Login fails with 500 error",
        description="I get a 500 Internal Server Error when I try to log in.",
        status="open",
        category="software"
    ),
    Ticket(
        title="Invoice not received",
        description="I haven't received my monthly invoice for March.",
        status="open",
        category="other"   # billing moved to "other"
    ),
    Ticket(
        title="Feature request: dark mode",
        description="Please add dark mode to the dashboard.",
        status="open",
        category="software"
    ),
    Ticket(
        title="App crashes on mobile",
        description="The mobile app crashes immediately after launch on iOS 17.",
        status="open",
        category="software"
    ),
    Ticket(
        title="Password reset email not arriving",
        description="I requested a password reset but the email never arrived.",
        status="open",
        category="access_permission"
    ),
]
SAMPLE_KB_ENTRIES = [
    KnowledgeBaseEntry(title="Resolving 500 Login Errors", content="Clear sessions, restart auth service, check DB connection.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="Invoice Re-delivery Process", content="Manually trigger invoice re-send from the billing portal.", category="other", source="knowledge_base"),
    KnowledgeBaseEntry(title="iOS App Crash Troubleshooting", content="Update to the latest app version and clear cache.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="Password Reset Email Delivery", content="Check spam folder, whitelist noreply@resolvex.ai, resend.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="Email Client Not Syncing", content="Check internet connection, refresh inbox, reconfigure email account.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="Software Update Failed", content="Check update server, restart application, reinstall update package.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="Application Freezing Frequently", content="Clear cache, update software, check system resources.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="File Not Opening in Application", content="Verify file format, update software, reinstall application.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="Error While Saving File", content="Check disk space, verify permissions, restart application.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="Plugin Not Working", content="Reinstall plugin, update application, check compatibility.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="Version Compatibility Issue", content="Upgrade/downgrade software version, check system requirements.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="Application Login Timeout", content="Check server response, restart app, verify credentials.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="Auto-Save Not Working", content="Enable auto-save settings, check disk permissions.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="Corrupted File Error", content="Restore backup, repair file using recovery tools.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="Mouse Not Detected", content="Reconnect mouse, change USB port, update drivers.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="System Random Shutdown", content="Check power supply, overheating, replace faulty hardware.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="Battery Not Charging", content="Check charger, replace battery, inspect charging port.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="Hard Disk Not Detected", content="Check SATA connection, BIOS settings, replace disk.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="USB Device Not Recognized", content="Reconnect device, update drivers, try different port.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="Fan Noise Issue", content="Clean dust, check fan alignment, replace if faulty.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="RAM Failure Issue", content="Reseat RAM, run memory diagnostics, replace module.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="Bluetooth Not Working", content="Enable Bluetooth, reinstall drivers, restart device.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="Touchpad Not Responding", content="Enable touchpad, update drivers, restart system.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="Camera Not Detected", content="Check drivers, enable camera settings, reinstall drivers.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="LAN Cable Not Working", content="Check cable, switch port, restart network device.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="IP Address Conflict", content="Release/renew IP, restart router, assign static IP.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="Network Dropping Frequently", content="Check signal strength, restart router, update firmware.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="Unable to Access Website", content="Check DNS, firewall settings, verify URL.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="Packet Loss Issue", content="Check network congestion, restart router, contact ISP.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="Port Not Accessible", content="Check firewall rules, open required port, restart service.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="VPN Disconnecting", content="Check internet stability, update VPN client.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="Router Not Responding", content="Restart router, reset settings, check power supply.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="Proxy Server Error", content="Check proxy settings, disable proxy if not needed.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="Network Authentication Failure", content="Verify credentials, reconnect network.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="Access Denied to Application", content="Check role permissions, grant access rights.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="User Cannot Reset Password", content="Check email delivery, resend reset link.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="Account Expired", content="Renew account access, update user profile.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="Login Loop Issue", content="Clear cookies, reset password, check session settings.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="Two-Factor Authentication Failure", content="Resync device, reset MFA settings.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="User Role Misconfigured", content="Assign correct role, update permissions.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="Unable to Access Dashboard", content="Check permissions, verify login status.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="Session Expired Frequently", content="Increase session timeout, check cookies.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="Access Revoked Error", content="Reassign permissions, verify user status.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="User Not Found in System", content="Check database entry, create user if missing.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="Multiple Failed Login Attempts", content="Lock account, notify user, enforce password reset.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="System Vulnerability Detected", content="Apply security patches, run vulnerability scan.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="Unauthorized File Access", content="Restrict permissions, audit logs.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="Security Certificate Expired", content="Renew SSL certificate, update configuration.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="Brute Force Attack Detected", content="Block IP, enable rate limiting.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="Data Breach Alert", content="Isolate system, notify security team, investigate logs.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="Firewall Misconfiguration", content="Review rules, correct settings.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="Suspicious File Download", content="Scan file, block source.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="Account Hijacking Suspected", content="Reset password, enable MFA.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="Encryption Failure Issue", content="Check encryption settings, reinstall certificates.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="Request for Account Upgrade", content="Verify eligibility, upgrade account plan.", category="other", source="knowledge_base"),
    KnowledgeBaseEntry(title="Service Downtime Inquiry", content="Check service status page, inform user.", category="other", source="knowledge_base"),
    KnowledgeBaseEntry(title="Feature Not Available", content="Inform user, log feature request.", category="other", source="knowledge_base"),
    KnowledgeBaseEntry(title="Training Request", content="Schedule training session, share materials.", category="other", source="knowledge_base"),
    KnowledgeBaseEntry(title="User Feedback Submission", content="Log feedback, forward to product team.", category="other", source="knowledge_base"),
    KnowledgeBaseEntry(title="Billing Discrepancy", content="Check invoice, verify charges.", category="other", source="knowledge_base"),
    KnowledgeBaseEntry(title="Refund Request", content="Verify transaction, initiate refund process.", category="other", source="knowledge_base"),
    KnowledgeBaseEntry(title="Service Activation Delay", content="Check backend process, escalate if needed.", category="other", source="knowledge_base"),
    KnowledgeBaseEntry(title="Documentation Request", content="Provide relevant documentation links.", category="other", source="knowledge_base"),
    KnowledgeBaseEntry(title="General Help Request", content="Provide support resources and FAQs.", category="other", source="knowledge_base"),
    KnowledgeBaseEntry(title="API Response Error", content="Check endpoint, verify API keys, restart service.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="Disk Space Full", content="Delete unnecessary files, extend storage.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="Cannot Connect to Server", content="Check network connectivity, restart server.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="User Cannot Upload Files", content="Check permissions, verify file size limits.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="Security Alert Notification", content="Investigate logs, take corrective action.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="Payment Gateway Timeout", content="Check network, retry transaction.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="External Device Not Working", content="Reconnect device, install drivers.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="Remote Desktop Not Connecting", content="Check network, enable RDP settings.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="User Cannot Access Email", content="Check credentials, reset password.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="Unauthorized API Access", content="Revoke API key, regenerate credentials.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="Application Update Loop", content="Clear update cache, restart app, reinstall latest version.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="Error Code 404 in App", content="Check endpoint URL, verify routing, restart service.", category="software    ", source="knowledge_base"),
    KnowledgeBaseEntry(title="Application Not Responding", content="Force close app, restart system, update software.", category="soft  ware", source="knowledge_base"),
    KnowledgeBaseEntry(title="Crash on Startup", content="Check logs, reinstall app, update dependencies.", category="  software", source="knowledge_base"),
    KnowledgeBaseEntry(title="Settings Not Saving", content="Check permissions, clear cache, restart application.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="Software License Expired", content="Renew license, update license key.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="API Integration Failure", content="Check API keys, verify endpoint, restart integration.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="File Upload Error", content="Check file size, format, server storage.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="Background Process Failure", content="Restart service, check logs, verify config.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="Data Sync Issue", content="Check connectivity, re-sync data, restart service.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="External Hard Drive Not Showing", content="Check connection, update drivers, assign drive letter.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="Speaker Not Working", content="Check volume settings, reinstall audio drivers.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="System Boot Failure", content="Check BIOS, boot order, repair OS.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="USB Port Not Working", content="Test with another device, update drivers.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="Screen Flickering Issue", content="Update display drivers, check cable.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="Battery Draining Fast", content="Close background apps, replace battery if needed.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="System Freezing Randomly", content="Check RAM, CPU usage, update drivers.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="Network Card Failure", content="Reinstall drivers, replace card if needed.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="Power Supply Failure", content="Check PSU, replace if faulty.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="Headphones Not Detected", content="Check jack, reinstall drivers.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="Unable to Ping Server", content="Check connectivity, firewall rules, server status.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="High Latency Issue", content="Check bandwidth usage, restart router.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="Router Firmware Issue", content="Update firmware, restart router.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="WiFi Signal Weak", content="Move closer to router, remove obstacles.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="Network Adapter Disabled", content="Enable adapter in settings, reinstall drivers.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="Cannot Access Shared Drive", content="Check network permissions, reconnect drive.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="Firewall Blocking Connection", content="Allow connection in firewall settings.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="VPN Authentication Error", content="Verify credentials, reset VPN profile.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="Network Loop Detected", content="Check switch configuration, disable loop.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="Gateway Not Reachable", content="Check router, restart network devices.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="User Cannot Access Email", content="Verify login credentials, reset password.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="Account Disabled", content="Enable account in admin panel.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="Access Token Expired", content="Refresh token, re-login user.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="Invalid Credentials Error", content="Reset password, verify username.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="User Locked Out of System", content="Unlock account, reset password.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="Missing Role Permissions", content="Assign correct role, update permissions.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="Login Redirect Loop", content="Clear cookies, check session config.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="Unauthorized API Access", content="Check API permissions, regenerate token.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="User Session Timeout", content="Increase timeout, refresh session.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="Multi-User Access Conflict", content="Check concurrent sessions, restrict access.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="Malicious Script Detected", content="Remove script, scan system, update security patches.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="Suspicious Network Activity", content="Analyze logs, block suspicious IP.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="Account Brute Force Attempt", content="Lock account, enable MFA.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="Data Leak Suspected", content="Investigate logs, secure data access.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="Expired Security Token", content="Generate new token, update authentication.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="Unauthorized Database Access", content="Restrict DB access, review logs.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="SSL Handshake Failure", content="Update certificates, check SSL config.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="Security Patch Missing", content="Apply latest patches, restart system.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="Phishing Attempt Detected", content="Block sender, notify users.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="Suspicious Login Location", content="Verify user, reset credentials.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="Service Request Delay", content="Check queue, escalate if needed.", category="other", source="knowledge_base"),
    KnowledgeBaseEntry(title="Request for Data Export", content="Verify request, generate export file.", category="other", source="knowledge_base"),
    KnowledgeBaseEntry(title="Subscription Upgrade Query", content="Provide plan details, upgrade account.", category="other", source="knowledge_base"),
    KnowledgeBaseEntry(title="User Complaint Handling", content="Log complaint, assign to support team.", category="other", source="knowledge_base"),
    KnowledgeBaseEntry(title="Training Material Request", content="Share documentation, schedule session.", category="other", source="knowledge_base"),
    KnowledgeBaseEntry(title="System Usage Inquiry", content="Provide usage guide, share tutorials.", category="other", source="knowledge_base"),
    KnowledgeBaseEntry(title="Billing Cycle Clarification", content="Explain billing cycle, share invoice.", category="other", source="knowledge_base"),
    KnowledgeBaseEntry(title="Account Deletion Request", content="Verify identity, process deletion.", category="other", source="knowledge_base"),
    KnowledgeBaseEntry(title="Support Escalation Request", content="Escalate ticket to higher level.", category="other", source="knowledge_base"),
    KnowledgeBaseEntry(title="Product Demo Request", content="Schedule demo session.", category="other", source="knowledge_base"),
    KnowledgeBaseEntry(title="Email Delivery Failure", content="Check SMTP server, verify email config.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="Disk Read Error", content="Run disk check, replace faulty disk.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="Server Not Reachable", content="Check network, restart server.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="User Cannot Download Files", content="Check permissions, verify network.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="Unauthorized Access Alert", content="Block user, review logs.", category="security", source="knowledge_base"),
    KnowledgeBaseEntry(title="Application Timeout Error", content="Increase timeout, check server load.", category="software", source="knowledge_base"),
    KnowledgeBaseEntry(title="Peripheral Device Failure", content="Reconnect device, update drivers.", category="hardware", source="knowledge_base"),
    KnowledgeBaseEntry(title="Network Bandwidth Limit Exceeded", content="Monitor usage, upgrade plan.", category="network", source="knowledge_base"),
    KnowledgeBaseEntry(title="Access Denied for API", content="Check permissions, update API key.", category="access_permission", source="knowledge_base"),
    KnowledgeBaseEntry(title="Security Breach Attempt", content="Isolate system, investigate logs.", category="security", source="knowledge_base")

        ]

SAMPLE_USERS = [
            User(name="Alice Agent",  email="alice@resolvex.ai",  role="agent"),
            User(name="Bob Admin",    email="bob@resolvex.ai",    role="admin"),
    ]


def seed():
    print("Initialising database...")
    init_db()

    db = SessionLocal()
    try:
        if db.query(Ticket).count() == 0:
            db.add_all(SAMPLE_TICKETS)
            print(f"Seeded {len(SAMPLE_TICKETS)} tickets.")

        existing_kb_titles = {kb.title for kb in db.query(KnowledgeBaseEntry.title).all()}
        new_kbs = [kb for kb in SAMPLE_KB_ENTRIES if kb.title not in existing_kb_titles]
        if new_kbs:
            db.add_all(new_kbs)
            print(f"Seeded {len(new_kbs)} new KB entries.")

        if db.query(User).count() == 0:
            db.add_all(SAMPLE_USERS)
            print(f"Seeded {len(SAMPLE_USERS)} users.")

        db.commit()
        print("✅ DB seed complete.")
    except Exception as exc:
        db.rollback()
        print(f"❌ Seed failed: {exc}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    seed()
