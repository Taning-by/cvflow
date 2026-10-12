"""Python 脚本的编辑控件：行号、当前行高亮、着色、按 Python 习惯缩进。

参数面板里的小编辑框和中央工作区的大编辑器用的是**同一个控件**，差别只有尺寸——
所以不管在哪儿写脚本，缩进、着色、Ctrl+S 应用的行为都一样。

着色自己写而不引 pygments：只认 Python，几十行就够，还能保证没有额外依赖。
"""
from __future__ import annotations

import keyword

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import (QColor, QFontMetricsF, QPainter, QSyntaxHighlighter, QTextCharFormat,
                           QTextCursor, QTextFormat)
from PySide6.QtWidgets import QPlainTextEdit, QTextEdit, QWidget

from .theme import C, mono_font

INDENT = "    "            # Python 用 4 个空格；编辑器里一律不插制表符

#: 着色用的颜色。浅底上保持足够对比度，和端口类型色是同一套取值习惯。
SYNTAX = {
    "keyword": "#9A1C7A",      # def / if / return…
    "builtin": "#1B6FA8",      # print / len / range…
    "string": "#3F7A34",
    "comment": "#8A97A3",
    "number": "#9A5B1E",
    "decorator": "#7044A0",
    "defname": "#115159",      # def / class 后面的名字
    "api": "#176B75",          # 节点脚本能用的对象（np、cv2、Image…）
}

#: 脚本里可以直接用的对象，见 PythonScript 的模板注释。单独着色，提示"这些是现成的"。
SCRIPT_API = ("np", "cv2", "Image", "Overlay", "Rect", "Point", "Line", "Circle", "NodeError",
              "ctx", "inputs", "params", "state")


def _fmt(color: str, bold: bool = False, italic: bool = False) -> QTextCharFormat:
    f = QTextCharFormat()
    f.setForeground(QColor(color))
    if bold:
        f.setFontWeight(600)
    if italic:
        f.setFontItalic(True)
    return f


class PythonHighlighter(QSyntaxHighlighter):
    """Python 着色。跨行的三引号字符串用 block state 记住，模板开头的文档字符串才不会串色。"""

    def __init__(self, document) -> None:
        super().__init__(document)
        self._kw = _fmt(SYNTAX["keyword"], bold=True)
        self._builtin = _fmt(SYNTAX["builtin"])
        self._api = _fmt(SYNTAX["api"], bold=True)
        self._str = _fmt(SYNTAX["string"])
        self._comment = _fmt(SYNTAX["comment"], italic=True)
        self._num = _fmt(SYNTAX["number"])
        self._deco = _fmt(SYNTAX["decorator"])
        self._defname = _fmt(SYNTAX["defname"], bold=True)
        self._keywords = set(keyword.kwlist) | {"match", "case"}
        self._builtins = set(dir(__builtins__) if isinstance(__builtins__, dict) else dir(__builtins__))
        self._api_names = set(SCRIPT_API)

    # ---- 逐行着色 ----
    def highlightBlock(self, text: str) -> None:
        start = 0
        state = self.previousBlockState()            # 1 / 2 ＝ 上一行结束时还在 ''' / """ 里
        if state in (1, 2):
            quote = "'''" if state == 1 else '"""'
            end = text.find(quote)
            if end < 0:
                self.setFormat(0, len(text), self._str)
                self.setCurrentBlockState(state)
                return
            self.setFormat(0, end + 3, self._str)
            start = end + 3
        self.setCurrentBlockState(0)
        i = start
        n = len(text)
        while i < n:
            ch = text[i]
            if ch == "#":                            # 注释吃到行尾
                self.setFormat(i, n - i, self._comment)
                return
            if text.startswith("'''", i) or text.startswith('"""', i):
                quote = text[i:i + 3]
                end = text.find(quote, i + 3)
                if end < 0:                          # 跨行：记住状态，下一行接着染
                    self.setFormat(i, n - i, self._str)
                    self.setCurrentBlockState(1 if quote == "'''" else 2)
                    return
                self.setFormat(i, end + 3 - i, self._str)
                i = end + 3
                continue
            if ch in "'\"":                          # 单行字符串，支持 \" 转义
                j = i + 1
                while j < n and text[j] != ch:
                    j += 2 if text[j] == "\\" else 1
                self.setFormat(i, min(j + 1, n) - i, self._str)
                i = j + 1
                continue
            if ch == "@" and (i == 0 or not text[i - 1].isalnum()):
                j = i + 1
                while j < n and (text[j].isalnum() or text[j] in "._"):
                    j += 1
                self.setFormat(i, j - i, self._deco)
                i = j
                continue
            if ch.isdigit() and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
                j = i
                while j < n and (text[j].isalnum() or text[j] in "._"):
                    j += 1
                self.setFormat(i, j - i, self._num)
                i = j
                continue
            if ch.isalpha() or ch == "_":
                j = i
                while j < n and (text[j].isalnum() or text[j] == "_"):
                    j += 1
                word = text[i:j]
                if word in self._keywords:
                    self.setFormat(i, j - i, self._kw)
                    if word in ("def", "class"):     # 紧跟着的名字也标出来
                        k = j
                        while k < n and text[k] == " ":
                            k += 1
                        m = k
                        while m < n and (text[m].isalnum() or text[m] == "_"):
                            m += 1
                        if m > k:
                            self.setFormat(k, m - k, self._defname)
                            j = m
                elif word in self._api_names:
                    self.setFormat(i, j - i, self._api)
                elif word in self._builtins:
                    self.setFormat(i, j - i, self._builtin)
                i = j
                continue
            i += 1


class _Gutter(QWidget):
    """行号栏。宽度跟着行数变，报错说"第 12 行"时能直接数到。"""

    def __init__(self, editor: "CodeEditor") -> None:
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self) -> QSize:
        return QSize(self._editor.gutter_width(), 0)

    def paintEvent(self, event) -> None:
        self._editor.paint_gutter(event)


class CodeEditor(QPlainTextEdit):
    """代码编辑框。

    * 行号 + 当前行高亮：几十行的脚本也能一眼定位
    * Tab / Shift+Tab 缩进（含整块选中），Enter 继承缩进、行尾是 ``:`` 再多缩一级
    * Ctrl+Enter 发出 ``apply_requested``（Ctrl+S 被主窗口的"保存方案"占着，到不了这里；
      保存方案时会自动先把编辑中的代码应用掉，所以按 Ctrl+S 也不会丢东西）
    * 永不自动换行——代码折行会把缩进读乱
    """

    apply_requested = Signal()

    def __init__(self, text: str = "", parent=None, font_size: int = 13) -> None:
        super().__init__(parent)
        self.setFont(mono_font(font_size))
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.setTabChangesFocus(False)
        self.setTabStopDistance(QFontMetricsF(self.font()).horizontalAdvance(" ") * len(INDENT))
        self.highlighter = PythonHighlighter(self.document())
        self._gutter = _Gutter(self)
        self.blockCountChanged.connect(lambda _: self._sync_gutter_width())
        self.updateRequest.connect(self._on_update_request)
        self.cursorPositionChanged.connect(self._highlight_current_line)
        self.setPlainText(text)
        self._sync_gutter_width()
        self._highlight_current_line()

    # ---- 行号栏 ----
    def gutter_width(self) -> int:
        digits = max(2, len(str(max(1, self.blockCount()))))
        return 12 + QFontMetricsF(self.font()).horizontalAdvance("9") * digits

    def _sync_gutter_width(self) -> None:
        self.setViewportMargins(int(self.gutter_width()), 0, 0, 0)

    def _on_update_request(self, rect: QRect, dy: int) -> None:
        if dy:
            self._gutter.scroll(0, dy)
        else:
            self._gutter.update(0, rect.y(), self._gutter.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._sync_gutter_width()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._gutter.setGeometry(QRect(cr.left(), cr.top(), int(self.gutter_width()), cr.height()))

    def paint_gutter(self, event) -> None:
        painter = QPainter(self._gutter)
        painter.fillRect(event.rect(), QColor(C["panel3"]))
        painter.setPen(QColor(C["border"]))
        painter.drawLine(event.rect().topRight(), event.rect().bottomRight())
        block = self.firstVisibleBlock()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        current = self.textCursor().blockNumber()
        painter.setFont(self.font())
        height = self.blockBoundingRect(block).height()
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and top + height >= event.rect().top():
                painter.setPen(QColor(C["accent"] if block.blockNumber() == current else C["faint"]))
                painter.drawText(0, int(top), self._gutter.width() - 6, int(height),
                                 Qt.AlignRight | Qt.AlignVCenter, str(block.blockNumber() + 1))
            block = block.next()
            top += height
            height = self.blockBoundingRect(block).height()

    def _highlight_current_line(self) -> None:
        sel = QTextEdit.ExtraSelection()
        sel.format.setBackground(QColor(C["accent_bg"]))
        sel.format.setProperty(QTextFormat.FullWidthSelection, True)
        sel.cursor = self.textCursor()
        sel.cursor.clearSelection()
        self.setExtraSelections([sel])
        self._gutter.update()

    # ---- 编辑习惯 ----
    def keyPressEvent(self, event) -> None:
        key, mods = event.key(), event.modifiers()
        ctrl = mods & Qt.ControlModifier
        if ctrl and key in (Qt.Key_S, Qt.Key_Return, Qt.Key_Enter):
            self.apply_requested.emit()
            event.accept()
            return
        if key == Qt.Key_Tab and not (mods & Qt.ShiftModifier):
            if self.textCursor().hasSelection():
                self._shift_lines(+1)
            else:
                self.insertPlainText(INDENT)
            event.accept()
            return
        if key == Qt.Key_Backtab or (key == Qt.Key_Tab and mods & Qt.ShiftModifier):
            self._shift_lines(-1)
            event.accept()
            return
        if key in (Qt.Key_Return, Qt.Key_Enter) and not ctrl:
            self._newline_keeping_indent()
            event.accept()
            return
        if key == Qt.Key_Backspace and not self.textCursor().hasSelection():
            cur = self.textCursor()
            col = cur.positionInBlock()
            before = cur.block().text()[:col]
            if col and not before.strip():          # 行首的空白：一次退一级缩进
                back = (col % len(INDENT)) or len(INDENT)
                for _ in range(min(back, col)):
                    cur.deletePreviousChar()
                event.accept()
                return
        super().keyPressEvent(event)

    def _newline_keeping_indent(self) -> None:
        cur = self.textCursor()
        line = cur.block().text()
        indent = line[:len(line) - len(line.lstrip())]
        if line[:cur.positionInBlock()].rstrip().endswith(":"):
            indent += INDENT
        cur.insertText("\n" + indent)
        self.setTextCursor(cur)

    def _shift_lines(self, direction: int) -> None:
        """把选中的（或光标所在的）整行往右/往左挪一级缩进。"""
        cur = self.textCursor()
        start, end = sorted((cur.selectionStart(), cur.selectionEnd()))
        cur.beginEditBlock()
        cur.setPosition(start)
        first = cur.blockNumber()
        cur.setPosition(end)
        last = cur.blockNumber()
        doc = self.document()
        for i in range(first, last + 1):
            block = doc.findBlockByNumber(i)
            text = block.text()
            edit = QTextCursor(block)
            if direction > 0:
                edit.insertText(INDENT)
            else:
                strip = len(text) - len(text.lstrip(" "))
                if strip:
                    edit.setPosition(block.position())
                    edit.setPosition(block.position() + min(len(INDENT), strip), QTextCursor.KeepAnchor)
                    edit.removeSelectedText()
        cur.endEditBlock()
        sel = QTextCursor(doc.findBlockByNumber(first))
        sel.setPosition(doc.findBlockByNumber(last).position()
                        + len(doc.findBlockByNumber(last).text()), QTextCursor.KeepAnchor)
        self.setTextCursor(sel)

    # ---- 供外部用 ----
    def cursor_position(self) -> tuple[int, int]:
        cur = self.textCursor()
        return cur.blockNumber() + 1, cur.positionInBlock() + 1

    def set_text_keeping_view(self, text: str) -> None:
        """换内容但尽量不动视口与光标（外部改了参数、需要刷新时用）。"""
        if text == self.toPlainText():
            return
        line, col = self.cursor_position()
        scroll = self.verticalScrollBar().value()
        self.setPlainText(text)
        cur = self.textCursor()
        block = self.document().findBlockByNumber(min(line - 1, self.blockCount() - 1))
        cur.setPosition(block.position() + min(col - 1, len(block.text())))
        self.setTextCursor(cur)
        self.verticalScrollBar().setValue(scroll)
