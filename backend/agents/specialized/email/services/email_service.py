# -*- coding: utf-8 -*-
"""
Email Service
Email operations and tools
"""

import smtplib
from email.message import EmailMessage
from email.utils import formataddr

from langchain_core.tools import tool

from app.core.config import settings


def _parse_recipients(to: str) -> list[str]:
    recipients = [addr.strip() for addr in to.replace(";", ",").split(",") if addr.strip()]
    return recipients


def _smtp_configured() -> bool:
    return bool(settings.smtp_host and settings.smtp_port and settings.smtp_user and settings.smtp_password)


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """
    使用系统统一配置的 SMTP 邮箱发送邮件。

    Args:
        to: 收件人邮箱，多个邮箱可用英文逗号或分号分隔
        subject: 邮件主题
        body: 邮件正文

    Returns:
        邮件发送结果
    """
    if not _smtp_configured():
        return "邮件发送失败：SMTP 未配置，请在 backend/.env 中设置 SMTP_HOST、SMTP_PORT、SMTP_USER、SMTP_PASSWORD。"

    recipients = _parse_recipients(to)
    if not recipients:
        return "邮件发送失败：收件人邮箱不能为空。"
    if not subject.strip():
        return "邮件发送失败：邮件主题不能为空。"
    if not body.strip():
        return "邮件发送失败：邮件正文不能为空。"

    msg = EmailMessage()
    msg["From"] = formataddr((settings.smtp_from_name, settings.smtp_user))
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject.strip()
    msg.set_content(body.strip(), subtype="plain", charset="utf-8")

    try:
        if settings.smtp_use_ssl:
            with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=20) as smtp:
                smtp.login(settings.smtp_user, settings.smtp_password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=20) as smtp:
                smtp.starttls()
                smtp.login(settings.smtp_user, settings.smtp_password)
                smtp.send_message(msg)
    except Exception as exc:
        return f"邮件发送失败：{exc}"

    return f"邮件已发送成功。收件人：{', '.join(recipients)}；主题：{subject.strip()}"


def get_email_tools():
    """
    Get list of email tools

    Returns:
        List of email tool functions
    """
    return [send_email]


# Agent metadata for supervisor
EMAIL_AGENT_INFO = {
    "name": "email_agent",
    "display_name": "邮件智能体",
    "description": "使用系统统一配置的 SMTP 邮箱发送邮件。发送前应让用户明确提供收件人、主题和正文。",
    "capabilities": [
        "发送邮件",
        "生成邮件正文",
        "检查邮件参数",
    ],
    "keywords": ["邮件", "email", "发送", "发邮件", "邮箱", "send"]
}
