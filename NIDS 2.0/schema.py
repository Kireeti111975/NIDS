"""
schema.py
---------
Defines the validated input schema for network traffic data using Pydantic.
All fields mirror the NSL-KDD / KDD Cup 99 feature conventions commonly used
in NIDS research.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enumerated categorical fields
# ---------------------------------------------------------------------------

class ProtocolType(str, Enum):
    tcp  = "tcp"
    udp  = "udp"
    icmp = "icmp"


class ServiceType(str, Enum):
    http        = "http"
    ftp         = "ftp"
    smtp        = "smtp"
    ssh         = "ssh"
    dns         = "dns"
    telnet      = "telnet"
    https       = "https"
    pop3        = "pop3"
    ftp_data    = "ftp_data"
    other       = "other"


class FlagType(str, Enum):
    SF  = "SF"
    S0  = "S0"
    REJ = "REJ"
    RSTO = "RSTO"
    SH  = "SH"
    S1  = "S1"
    S2  = "S2"
    S3  = "S3"
    OTH = "OTH"


# ---------------------------------------------------------------------------
# Main traffic record schema
# ---------------------------------------------------------------------------

class TrafficRecord(BaseModel):
    """A single network connection record submitted for classification."""

    duration:      Annotated[int,   Field(ge=0,  description="Duration of the connection in seconds")]
    protocol_type: ProtocolType                                                  = Field(..., description="Network protocol (tcp | udp | icmp)")
    service:       ServiceType                                                   = Field(..., description="Destination network service")
    flag:          FlagType                                                      = Field(..., description="Connection status flag")
    src_bytes:     Annotated[int,   Field(ge=0,  description="Bytes sent from source to destination")]
    dst_bytes:     Annotated[int,   Field(ge=0,  description="Bytes sent from destination to source")]
    count:         Annotated[int,   Field(ge=0,  description="Connections to same host in the past 2 seconds")]
    srv_count:     Annotated[int,   Field(ge=0,  description="Connections to same service in the past 2 seconds")]
    serror_rate:   Annotated[float, Field(ge=0.0, le=1.0, description="% of connections with SYN errors")]
    rerror_rate:   Annotated[float, Field(ge=0.0, le=1.0, description="% of connections with REJ errors")]
    same_srv_rate: Annotated[float, Field(ge=0.0, le=1.0, description="% of connections to the same service")]
    diff_srv_rate: Annotated[float, Field(ge=0.0, le=1.0, description="% of connections to different services")]

    @model_validator(mode="after")
    def src_or_dst_bytes_nonzero_for_tcp(self) -> "TrafficRecord":
        """Warn-level guard: TCP connections should carry at least some bytes."""
        if self.protocol_type == ProtocolType.tcp:
            if self.src_bytes == 0 and self.dst_bytes == 0 and self.flag == FlagType.SF:
                raise ValueError(
                    "A completed TCP connection (flag=SF) should have non-zero byte counts."
                )
        return self
