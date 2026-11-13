from __future__ import annotations

import pytest
from pydantic import ValidationError

from sentinel.tools.nmap import DiscoverArgs, parse_nmap_xml
from sentinel.tools.http import ContentDiscoveryArgs, FetchArgs
from sentinel.tools.nuclei import NucleiScanArgs
from sentinel.tools.base import default_registry

SAMPLE_NMAP_XML = """<?xml version="1.0"?>
<nmaprun>
<host>
<status state="up"/>
<address addr="10.10.10.5" addrtype="ipv4"/>
<ports>
<port protocol="tcp" portid="22">
<state state="open"/>
<service name="ssh" product="OpenSSH" version="8.2p1"/>
</port>
<port protocol="tcp" portid="80">
<state state="open"/>
<service name="http" product="Apache httpd" version="2.4.41"/>
</port>
</ports>
</host>
</nmaprun>
"""


def test_nmap_discover_rejects_shell_metacharacters():
    with pytest.raises(ValidationError):
        DiscoverArgs(target="10.10.10.5; rm -rf /")


def test_nmap_discover_accepts_valid_ip():
    args = DiscoverArgs(target="10.10.10.5")
    assert args.target == "10.10.10.5"


def test_http_fetch_rejects_non_http_scheme():
    with pytest.raises(ValidationError):
        FetchArgs(url="file:///etc/passwd")


def test_http_content_discovery_rejects_bad_wordlist_path():
    with pytest.raises(ValidationError):
        ContentDiscoveryArgs(url="http://10.10.10.5/", wordlist="$(cat /etc/shadow)")


def test_nuclei_enforces_approved_tag_allowlist():
    with pytest.raises(ValidationError):
        NucleiScanArgs(url="http://10.10.10.5/", tags="rce,exploit")
    ok = NucleiScanArgs(url="http://10.10.10.5/", tags="tech,exposures")
    assert "tech" in ok.tags


def test_parse_nmap_xml_extracts_services():
    result = parse_nmap_xml(SAMPLE_NMAP_XML)
    assert len(result["hosts"]) == 1
    host = result["hosts"][0]
    assert host["address"] == "10.10.10.5"
    assert len(host["ports"]) == 2
    ssh_port = next(p for p in host["ports"] if p["port"] == 22)
    assert ssh_port["service_name"] == "ssh"
    assert ssh_port["product"] == "OpenSSH"


def test_parse_nmap_xml_handles_malformed_input_gracefully():
    result = parse_nmap_xml("not xml at all <<<")
    assert result["hosts"] == []
    assert "parse_error" in result


def test_registry_has_expected_tools():
    registry = default_registry()
    names = {t.name for t in registry.all()}
    assert {"nmap", "http", "ssh", "ftp", "smb", "dns", "nuclei", "zap"}.issubset(names)


def test_resolve_unknown_tool_raises():
    registry = default_registry()
    with pytest.raises(KeyError):
        registry.resolve("doesnotexist", "op")


def test_resolve_unknown_operation_raises():
    registry = default_registry()
    with pytest.raises(KeyError):
        registry.resolve("nmap", "doesnotexist")
