from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QMenu, QTreeWidget, QTreeWidgetItem

KIND_PREFIX = {"mesh": "▣ ", "light": "☀ ", "group": "▤ "}


class Outliner(QTreeWidget):
    selectionEdited = Signal(list)      # el usuario cambió la selección
    renamed = Signal()
    deleteRequested = Signal()
    groupRequested = Signal()

    def __init__(self, scene):
        super().__init__()
        self.scene = scene
        self.setHeaderHidden(True)
        self.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.itemSelectionChanged.connect(self._on_sel)
        self.itemChanged.connect(self._on_rename)
        self._items = {}
        self._block = False

    def refresh(self):
        self._block = True
        self.clear(); self._items = {}
        def add(parent_item, pid):
            for n in self.scene.children(pid):
                it = QTreeWidgetItem([KIND_PREFIX.get(n.kind, "") + n.name])
                it.setData(0, Qt.ItemDataRole.UserRole, n.id)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsEditable)
                (parent_item.addChild(it) if parent_item else self.addTopLevelItem(it))
                self._items[n.id] = it
                add(it, n.id)
        add(None, None)
        self.expandAll()
        self._block = False
        self.sync_selection()

    def sync_selection(self):
        self._block = True
        self.clearSelection()
        for i in self.scene.selection:
            if i in self._items:
                self._items[i].setSelected(True)
        self._block = False

    def _on_sel(self):
        if not self._block:
            self.selectionEdited.emit([it.data(0, Qt.ItemDataRole.UserRole) for it in self.selectedItems()])

    def _on_rename(self, item, _col):
        if self._block:
            return
        n = self.scene.nodes.get(item.data(0, Qt.ItemDataRole.UserRole))
        if n:
            n.name = item.text(0).lstrip("▣☀▤ ").strip() or n.name
            self.renamed.emit()

    def _menu(self, pos):
        m = QMenu(self)
        m.addAction("Group (Ctrl+G)", self.groupRequested.emit)
        m.addAction("Delete", self.deleteRequested.emit)
        m.exec(self.viewport().mapToGlobal(pos))
