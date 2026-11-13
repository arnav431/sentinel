"""Seed training examples for Sentinel Intelligence (Part 2).

Every example here is **synthetic** — invented scenarios and invented tool
output, written to be representative of what `sentinel -a` would actually
produce, not copied from a real walkthrough, real flag, or memorized attack
path. `synthetic: True` is set on every record and `prepare_dataset.py`
refuses to emit a record that doesn't declare it.

Each example is a (context, target_decision) pair:
  - `context` is the kind of dict `planner.build_context()` produces: the
    assessment so far (services, hypotheses, findings, completed
    operations, available operations) plus, for result-interpretation
    examples, the most recent tool invocation's output.
  - `target_decision` is a `StructuredDecision`-shaped dict — the label.

`scenario_id` groups examples that share a fictional "machine" so
`prepare_dataset.py` can split train/val/test by scenario and avoid
leaking near-duplicate states across the split boundary.
"""
from __future__ import annotations

EXAMPLES: list[dict] = [
    # ---- scenario: "lab-alpha" — fresh recon through an exposed SMB share
    {
        "scenario_id": "lab-alpha",
        "category": "initial_recon_planning",
        "synthetic": True,
        "context": {
            "assessment_id": "assess_demo_alpha",
            "target": "10.10.10.201",
            "phase": "recon",
            "services": [],
            "hypotheses": [],
            "findings": [],
            "completed_operations": [],
            "available_operations": [
                {"tool_name": "nmap", "operation": "discover", "description": "Fast host/port discovery", "risk": "safe"},
                {"tool_name": "nmap", "operation": "service_scan", "description": "Service/version detection", "risk": "safe"},
            ],
        },
        "target_decision": {
            "assessment_id": "assess_demo_alpha",
            "observations": ["No services discovered yet; this is the first step of the assessment."],
            "hypotheses": [],
            "missing_information": ["Which ports are open on the target", "What services are listening"],
            "proposed_action": {"tool": "nmap.discover", "arguments": {"target": "10.10.10.201"}},
            "rationale": "No scan has been run yet. A fast port discovery is the lowest-cost way to learn what's worth investigating before committing to slower, more detailed checks.",
            "expected_evidence": ["A list of open TCP ports on 10.10.10.201"],
            "stop_condition": "none",
        },
    },
    {
        "scenario_id": "lab-alpha",
        "category": "nmap_interpretation",
        "synthetic": True,
        "context": {
            "assessment_id": "assess_demo_alpha",
            "tool": "nmap", "operation": "discover", "exit_code": 0,
            "stdout_excerpt": (
                "Nmap scan report for 10.10.10.201\n"
                "PORT    STATE SERVICE\n"
                "22/tcp  open  ssh\n"
                "139/tcp open  netbios-ssn\n"
                "445/tcp open  microsoft-ds\n"
            ),
        },
        "target_decision": {
            "assessment_id": "assess_demo_alpha",
            "observations": [
                "Port 22/tcp (ssh) is open.",
                "Ports 139/tcp and 445/tcp are open, consistent with an SMB service.",
            ],
            "hypotheses": [
                "The host may expose SMB shares that are readable without authentication.",
            ],
            "missing_information": ["SMB share names and permissions", "Whether anonymous/guest SMB access is allowed"],
            "proposed_action": {"tool": "smb.list_shares_anonymous", "arguments": {"host": "10.10.10.201"}},
            "rationale": "SMB ports being open doesn't by itself tell us anything exploitable — the next cheapest, non-intrusive check is whether anonymous enumeration is even permitted.",
            "expected_evidence": ["A list of SMB shares, or an authentication failure indicating anonymous access is disabled"],
            "stop_condition": "none",
        },
    },
    {
        "scenario_id": "lab-alpha",
        "category": "smb_enumeration_interpretation",
        "synthetic": True,
        "context": {
            "assessment_id": "assess_demo_alpha",
            "tool": "smb", "operation": "list_shares_anonymous", "exit_code": 0,
            "stdout_excerpt": (
                "Sharename       Type      Comment\n"
                "---------       ----      -------\n"
                "ADMIN$          Disk      Remote Admin\n"
                "C$              Disk      Default share\n"
                "Notes           Disk      Team notes\n"
                "IPC$            IPC       Remote IPC\n"
            ),
        },
        "target_decision": {
            "assessment_id": "assess_demo_alpha",
            "observations": [
                "Anonymous SMB session succeeded and listed shares.",
                "A non-default share named 'Notes' is present alongside the standard administrative shares.",
            ],
            "hypotheses": [
                "The 'Notes' share may be readable anonymously and could contain information useful for further enumeration (e.g. usernames, hints, credentials left in plaintext).",
            ],
            "missing_information": ["Whether the 'Notes' share is actually readable (listing shares does not confirm read access)", "What files, if any, it contains"],
            "proposed_action": None,
            "rationale": "Listing shares only confirms they exist, not that they're readable. No registered operation currently performs a safe, read-only directory listing of an individual SMB share's contents, so this needs a registered tool capability that doesn't yet exist rather than a guess.",
            "expected_evidence": [],
            "stop_condition": "requires_user_input",
        },
    },
    # ---- scenario: "lab-bravo" — web app with a misleading banner
    {
        "scenario_id": "lab-bravo",
        "category": "http_investigation",
        "synthetic": True,
        "context": {
            "assessment_id": "assess_demo_bravo",
            "tool": "http", "operation": "fetch_headers", "exit_code": 0,
            "stdout_excerpt": (
                "HTTP/1.1 200 OK\n"
                "Server: Apache/2.2.8 (Ubuntu)\n"
                "X-Powered-By: PHP/5.2.4\n"
                "Content-Type: text/html; charset=UTF-8\n"
            ),
        },
        "target_decision": {
            "assessment_id": "assess_demo_bravo",
            "observations": [
                "The Server header reports Apache 2.2.8 and PHP 5.2.4, both long past end-of-life.",
            ],
            "hypotheses": [
                "The server may genuinely be running an old, vulnerable stack — but Server/X-Powered-By headers are attacker- and admin-editable and frequently spoofed on CTF-style boxes, so this is not yet reliable evidence of the actual software version.",
            ],
            "missing_information": ["Whether the reported version is genuine", "What web technologies are actually detectable through content fingerprinting"],
            "proposed_action": {"tool": "http.fingerprint", "arguments": {"url": "http://10.10.10.210/"}},
            "rationale": "A banner alone is not reliable evidence of the real software version. Fingerprinting the actual page content provides an independent signal before treating the banner's claimed version as fact.",
            "expected_evidence": ["Detected web technologies/CMS fingerprints independent of the Server header"],
            "stop_condition": "none",
        },
    },
    {
        "scenario_id": "lab-bravo",
        "category": "distinguishing_possible_from_verified",
        "synthetic": True,
        "context": {
            "assessment_id": "assess_demo_bravo",
            "tool": "nuclei", "operation": "scan_approved_templates", "exit_code": 0,
            "stdout_excerpt": (
                '{"template-id":"exposed-panel-generic","host":"http://10.10.10.210/admin",'
                '"matched-at":"http://10.10.10.210/admin","info":{"severity":"info"}}\n'
            ),
        },
        "target_decision": {
            "assessment_id": "assess_demo_bravo",
            "observations": [
                "An automated template matched an exposed admin panel at /admin.",
            ],
            "hypotheses": [
                "An administrative interface is reachable at /admin and may be worth manual review for default credentials or further misconfiguration.",
            ],
            "missing_information": ["Whether the panel actually requires meaningful authentication", "Whether any credentials work"],
            "proposed_action": None,
            "rationale": "This is a scanner alert, not a confirmed finding — it indicates a URL pattern matched, not that the panel is exploitable or misconfigured. No credential-testing operation is registered or appropriate here without explicit authorization and a safe, non-brute-force method, so this should be surfaced to the user as an unverified observation rather than escalated automatically.",
            "expected_evidence": [],
            "stop_condition": "requires_user_input",
        },
    },
    # ---- scenario: "lab-charlie" — failed hypothesis, needs a pivot
    {
        "scenario_id": "lab-charlie",
        "category": "pivot_after_failed_hypothesis",
        "synthetic": True,
        "context": {
            "assessment_id": "assess_demo_charlie",
            "target": "10.10.10.220",
            "phase": "investigation",
            "services": [
                {"host": "10.10.10.220", "port": 21, "service_name": "ftp"},
                {"host": "10.10.10.220", "port": 80, "service_name": "http"},
            ],
            "hypotheses": [
                {"hypothesis_id": "hyp_001", "description": "FTP allows anonymous login.", "status": "contradicted"},
            ],
            "findings": [],
            "completed_operations": [{"tool": "ftp", "operation": "banner_and_anon_check", "exit_code": 0}],
            "available_operations": [
                {"tool_name": "http", "operation": "content_discovery", "description": "Directory/file discovery", "risk": "intrusive"},
                {"tool_name": "http", "operation": "fingerprint", "description": "Identify web technologies", "risk": "safe"},
            ],
        },
        "target_decision": {
            "assessment_id": "assess_demo_charlie",
            "observations": ["The anonymous-FTP hypothesis was tested and contradicted — anonymous login is refused."],
            "hypotheses": ["FTP is not currently a promising lead; port 80 (HTTP) remains unexplored and should be investigated next."],
            "missing_information": ["What web technologies are running on port 80", "Whether any content discovery is warranted"],
            "proposed_action": {"tool": "http.fingerprint", "arguments": {"url": "http://10.10.10.220/"}},
            "rationale": "The FTP hypothesis was disproven, so continuing to probe FTP would be a redundant/unproductive action. Port 80 is open and unexamined — fingerprinting it first is the least intrusive next step before considering content discovery.",
            "expected_evidence": ["Detected web technologies on port 80"],
            "stop_condition": "none",
        },
    },
    {
        "scenario_id": "lab-charlie",
        "category": "avoiding_redundant_actions",
        "synthetic": True,
        "context": {
            "assessment_id": "assess_demo_charlie",
            "target": "10.10.10.220",
            "phase": "investigation",
            "services": [{"host": "10.10.10.220", "port": 80, "service_name": "http"}],
            "hypotheses": [],
            "findings": [],
            "completed_operations": [
                {"tool": "http", "operation": "fingerprint", "exit_code": 0},
                {"tool": "http", "operation": "fingerprint", "exit_code": 0},
            ],
            "available_operations": [
                {"tool_name": "http", "operation": "fingerprint", "description": "Identify web technologies", "risk": "safe"},
                {"tool_name": "http", "operation": "content_discovery", "description": "Directory/file discovery", "risk": "intrusive"},
            ],
        },
        "target_decision": {
            "assessment_id": "assess_demo_charlie",
            "observations": ["Fingerprinting has already been run twice against the same URL with no new information between runs."],
            "hypotheses": [],
            "missing_information": ["Directory/file structure of the web application"],
            "proposed_action": {"tool": "http.content_discovery", "arguments": {"url": "http://10.10.10.220/"}},
            "rationale": "Re-running fingerprinting a third time would not produce new evidence. Content discovery is the next reasonable step but is intrusive, so it requires explicit user approval before it runs.",
            "expected_evidence": ["Discovered directories/files not linked from the homepage"],
            "stop_condition": "none",
        },
    },
    # ---- scenario: "lab-delta" — scope violation suspected
    {
        "scenario_id": "lab-delta",
        "category": "scope_violation_recognition",
        "synthetic": True,
        "context": {
            "assessment_id": "assess_demo_delta",
            "target": "10.10.10.230",
            "phase": "investigation",
            "services": [{"host": "10.10.10.230", "port": 80, "service_name": "http"}],
            "hypotheses": ["The application may proxy or redirect to an internal admin host at 10.10.11.5."],
            "findings": [],
            "completed_operations": [{"tool": "http", "operation": "fetch_headers", "exit_code": 0}],
            "available_operations": [
                {"tool_name": "nmap", "operation": "discover", "description": "Fast host/port discovery", "risk": "safe"},
            ],
        },
        "target_decision": {
            "assessment_id": "assess_demo_delta",
            "observations": ["A response header suggested a redirect target of 10.10.11.5, which is outside the authorized target 10.10.10.230."],
            "hypotheses": ["The redirect target may be an internal host unrelated to the authorized scope."],
            "missing_information": ["Whether 10.10.11.5 is in authorized scope"],
            "proposed_action": None,
            "rationale": "Scanning or following the redirect to 10.10.11.5 would mean testing a host that was never added to scope. This must be flagged to the user rather than acted on automatically, even though the lead looks promising.",
            "expected_evidence": [],
            "stop_condition": "scope_violation_suspected",
        },
    },
    # ---- scenario: "lab-echo" — sufficient evidence to stop
    {
        "scenario_id": "lab-echo",
        "category": "stop_condition_sufficient_evidence",
        "synthetic": True,
        "context": {
            "assessment_id": "assess_demo_echo",
            "target": "10.10.10.240",
            "phase": "verification",
            "services": [{"host": "10.10.10.240", "port": 21, "service_name": "ftp"}],
            "hypotheses": [{"hypothesis_id": "hyp_010", "description": "FTP allows anonymous login and the root is writable.", "status": "verified"}],
            "findings": [{"finding_id": "find_010", "title": "Anonymous FTP with writable root", "verification_status": "confirmed"}],
            "completed_operations": [
                {"tool": "ftp", "operation": "banner_and_anon_check", "exit_code": 0},
            ],
            "available_operations": [],
        },
        "target_decision": {
            "assessment_id": "assess_demo_echo",
            "observations": ["Anonymous FTP login was confirmed and the share was confirmed writable; a finding has already been recorded with confirmed verification status."],
            "hypotheses": [],
            "missing_information": [],
            "proposed_action": None,
            "rationale": "The objective for this service (determine whether anonymous FTP exposes writable access) has been met and recorded as a confirmed finding. No further registered checks are pending for this host.",
            "expected_evidence": [],
            "stop_condition": "objective_met",
        },
    },
    # ---- scenario: "lab-foxtrot" — DNS + privilege-escalation-lead framing
    {
        "scenario_id": "lab-foxtrot",
        "category": "dns_enumeration",
        "synthetic": True,
        "context": {
            "assessment_id": "assess_demo_foxtrot",
            "tool": "dns", "operation": "lookup", "exit_code": 0,
            "stdout_excerpt": (
                ";; ANSWER SECTION:\n"
                "intranet.lab.htb.   300  IN  A  10.10.10.250\n"
            ),
        },
        "target_decision": {
            "assessment_id": "assess_demo_foxtrot",
            "observations": ["A DNS A record resolves intranet.lab.htb to 10.10.10.250, the same host already in scope."],
            "hypotheses": ["The target may serve virtual-host-routed content under the hostname intranet.lab.htb rather than only its bare IP."],
            "missing_information": ["What content, if any, is served when requests use the Host header intranet.lab.htb"],
            "proposed_action": {"tool": "http.fetch_headers", "arguments": {"url": "http://intranet.lab.htb/"}},
            "rationale": "Since intranet.lab.htb resolves to an already-in-scope IP, checking it via HTTP is still within the authorized target and may reveal virtual-host-specific content not visible on the bare IP.",
            "expected_evidence": ["Response headers for the virtual host, potentially differing from the bare-IP response"],
            "stop_condition": "none",
        },
    },
    {
        "scenario_id": "lab-foxtrot",
        "category": "privilege_escalation_lead",
        "synthetic": True,
        "context": {
            "assessment_id": "assess_demo_foxtrot",
            "note": "User-supplied authorized linPEAS-style enumeration output pasted into chat for interpretation (not run by Sentinel directly — no linPEAS adapter is registered).",
            "stdout_excerpt": (
                "[+] Sudo rights\n"
                "Matching Defaults entries for www-data on lab:\n"
                "User www-data may run the following commands on lab:\n"
                "    (root) NOPASSWD: /usr/bin/python3 /opt/scripts/backup.py\n"
            ),
        },
        "target_decision": {
            "assessment_id": "assess_demo_foxtrot",
            "observations": [
                "The www-data user can run /opt/scripts/backup.py as root without a password via sudo.",
            ],
            "hypotheses": [
                "If /opt/scripts/backup.py is writable by www-data or imports an attacker-writable module, this sudo rule could allow privilege escalation to root.",
            ],
            "missing_information": [
                "Whether /opt/scripts/backup.py (and anything it imports) is writable by www-data",
                "The actual contents and behavior of backup.py",
            ],
            "proposed_action": None,
            "rationale": "This is a credible, well-known privilege-escalation pattern (NOPASSWD sudo rule on an attacker-influenceable script), but confirming it requires checking file permissions on the target host, which is outside Sentinel's currently registered read-only network-enumeration tools. This should be surfaced to the user as a lead to investigate manually/interactively, not auto-escalated.",
            "expected_evidence": [],
            "stop_condition": "requires_user_input",
        },
    },
    # ---- scenario: "lab-golf" — inconclusive check
    {
        "scenario_id": "lab-golf",
        "category": "handling_inconclusive_checks",
        "synthetic": True,
        "context": {
            "assessment_id": "assess_demo_golf",
            "tool": "ssh", "operation": "banner_grab", "exit_code": 0,
            "stdout_excerpt": (
                "Host script results:\n"
                "|_ssh-hostkey: ERROR: Script execution failed (use -d to debug)\n"
            ),
        },
        "target_decision": {
            "assessment_id": "assess_demo_golf",
            "observations": ["The SSH algorithm/hostkey enumeration script failed to execute and returned no usable data."],
            "hypotheses": [],
            "missing_information": ["SSH server version and supported algorithms"],
            "proposed_action": None,
            "rationale": "The check was inconclusive rather than negative — it doesn't support or contradict any hypothesis about the SSH service, it simply didn't produce evidence. Re-running the identical command immediately would likely fail the same way; this should be reported as inconclusive rather than treated as either a pass or a finding.",
            "expected_evidence": [],
            "stop_condition": "requires_user_input",
        },
    },
    # ---- scenario: "lab-hotel" — budget exhausted
    {
        "scenario_id": "lab-hotel",
        "category": "stop_condition_budget_exhausted",
        "synthetic": True,
        "context": {
            "assessment_id": "assess_demo_hotel",
            "target": "10.10.10.260",
            "phase": "investigation",
            "iteration_count": 40,
            "max_iterations": 40,
            "services": [{"host": "10.10.10.260", "port": 80, "service_name": "http"}],
            "hypotheses": [{"hypothesis_id": "hyp_020", "description": "Possible SQL injection in login form.", "status": "unresolved"}],
            "findings": [],
            "completed_operations": [],
            "available_operations": [
                {"tool_name": "http", "operation": "content_discovery", "description": "Directory/file discovery", "risk": "intrusive"},
            ],
        },
        "target_decision": {
            "assessment_id": "assess_demo_hotel",
            "observations": ["The assessment has used 40 of its 40 allotted iterations."],
            "hypotheses": ["An unresolved SQL-injection hypothesis remains open and was not reached before the budget ran out."],
            "missing_information": ["Verification of the SQL-injection hypothesis"],
            "proposed_action": None,
            "rationale": "The iteration budget is exhausted. Continuing would require the user to explicitly raise the limit; proposing another tool call here would be proposing an action the policy engine will refuse to run.",
            "expected_evidence": [],
            "stop_condition": "budget_exhausted",
        },
    },
    # ---- scenario: "lab-india" — correlating multiple services into one hypothesis
    {
        "scenario_id": "lab-india",
        "category": "evidence_correlation",
        "synthetic": True,
        "context": {
            "assessment_id": "assess_demo_india",
            "target": "10.10.10.270",
            "phase": "investigation",
            "services": [
                {"host": "10.10.10.270", "port": 80, "service_name": "http", "product": "Apache", "version": "2.4.49"},
                {"host": "10.10.10.270", "port": 22, "service_name": "ssh", "product": "OpenSSH", "version": "8.2p1"},
            ],
            "hypotheses": [],
            "findings": [],
            "completed_operations": [{"tool": "nmap", "operation": "service_scan", "exit_code": 0}],
            "available_operations": [
                {"tool_name": "nuclei", "operation": "scan_approved_templates", "description": "Approved low-risk template checks", "risk": "safe"},
            ],
        },
        "target_decision": {
            "assessment_id": "assess_demo_india",
            "observations": ["Service/version detection reports Apache httpd 2.4.49."],
            "hypotheses": ["Apache 2.4.49 is a version associated with a known path-traversal/RCE advisory (CVE-2021-41773/CVE-2021-42013); this should be checked against advisory detail rather than assumed vulnerable, since distro backporting can mean the reported version string doesn't reflect the patch level."],
            "missing_information": ["Whether this specific build has the patch backported", "Independent confirmation via a non-intrusive template check"],
            "proposed_action": {"tool": "nuclei.scan_approved_templates", "arguments": {"url": "http://10.10.10.270/", "tags": "exposures,tech"}},
            "rationale": "A version number alone is not proof of vulnerability, especially with common backporting practices. An approved, non-intrusive template scan is a reasonable next step to gather independent, low-risk evidence before treating this as a confirmed lead.",
            "expected_evidence": ["Any matched low-risk exposures/tech-fingerprint templates"],
            "stop_condition": "none",
        },
    },
    # ---- scenario: "lab-juliet" — writing an evidence-grounded finding + remediation
    {
        "scenario_id": "lab-juliet",
        "category": "evidence_grounded_finding_and_remediation",
        "synthetic": True,
        "context": {
            "assessment_id": "assess_demo_juliet",
            "tool": "smb", "operation": "enum_shares_users", "exit_code": 0,
            "stdout_excerpt": (
                "[*] Enumerating users via 'lsaquery'\n"
                "S-1-5-21-...-1001 svc_backup (Local User)\n"
                "S-1-5-21-...-1002 j.admin (Local User)\n"
                "[*] Null session allowed\n"
            ),
        },
        "target_decision": {
            "assessment_id": "assess_demo_juliet",
            "observations": [
                "A null (unauthenticated) SMB session was permitted and returned local usernames: svc_backup, j.admin.",
            ],
            "hypotheses": [
                "Null-session username enumeration gives an attacker a username list for credential-guessing or targeted phishing against this host.",
            ],
            "missing_information": [],
            "proposed_action": None,
            "rationale": "Null-session enumeration succeeding and returning real account names is itself a confirmed, verifiable finding — not just a hypothesis — since it was directly observed rather than inferred.",
            "expected_evidence": [],
            "stop_condition": "none",
        },
        "finding_label": {
            "title": "SMB null session permits username enumeration",
            "severity": "low",
            "confidence": 0.95,
            "verification_status": "confirmed",
            "remediation": "Disable null/anonymous SMB sessions (RestrictAnonymous / RestrictAnonymousSAM registry settings, or equivalent Samba 'restrict anonymous' setting) so unauthenticated clients cannot enumerate local accounts.",
        },
    },
]
