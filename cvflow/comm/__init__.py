"""通信子系统：设备、分帧、解析/格式化、Modbus 数据点、触发规则与检测握手。"""
from .base import CommDevice, CommError, Peer
from .framing import FramingConfig, Framer, FRAMING_KINDS, parse_hex, to_hex, unescape
from .handshake import ERROR_CODES, ERROR_TEXT, Handshake, HandshakeConfig
from .manager import (ACTIONS, DATA_MATCHES, DEVICE_KIND_LABELS, DEVICE_KINDS, DEVICE_STATUS,
                      PLANNED_PROTOCOLS, REGISTER_MATCHES, RULE_SOURCES, SEND_TARGETS,
                      TEXT_MATCHES, WHEN_CHOICES, CommLog, CommManager, MessageInbox,
                      ReceiveRule, SendRule, format_template, get_manager, set_manager)
from .modbus_map import AREAS, DataPoint, DataPointTable, DTYPES, LAYOUTS, MapError
from .parsing import FormatField, FormatRule, ParseError, ParseField, ParseRule

__all__ = [
    "CommDevice", "CommError", "Peer",
    "FramingConfig", "Framer", "FRAMING_KINDS", "parse_hex", "to_hex", "unescape",
    "Handshake", "HandshakeConfig", "ERROR_CODES", "ERROR_TEXT",
    "CommManager", "CommLog", "MessageInbox", "ReceiveRule", "SendRule",
    "format_template", "get_manager", "set_manager",
    "DEVICE_KINDS", "DEVICE_KIND_LABELS", "DEVICE_STATUS", "PLANNED_PROTOCOLS",
    "TEXT_MATCHES", "DATA_MATCHES", "REGISTER_MATCHES", "RULE_SOURCES", "ACTIONS",
    "SEND_TARGETS", "WHEN_CHOICES",
    "DataPoint", "DataPointTable", "AREAS", "DTYPES", "LAYOUTS", "MapError",
    "ParseRule", "ParseField", "FormatRule", "FormatField", "ParseError",
]
