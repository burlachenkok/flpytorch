# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './../forms/SimulationWidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_SimulationWidget(object):
    def setupUi(self, SimulationWidget):
        SimulationWidget.setObjectName("SimulationWidget")
        SimulationWidget.resize(1133, 639)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(SimulationWidget.sizePolicy().hasHeightForWidth())
        SimulationWidget.setSizePolicy(sizePolicy)
        SimulationWidget.setStyleSheet("")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(SimulationWidget)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.splitter = QtWidgets.QSplitter(SimulationWidget)
        self.splitter.setStyleSheet("")
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setHandleWidth(6)
        self.splitter.setObjectName("splitter")
        self.layoutWidget = QtWidgets.QWidget(self.splitter)
        self.layoutWidget.setObjectName("layoutWidget")
        self.loVertical = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.loVertical.setContentsMargins(0, 0, 0, 0)
        self.loVertical.setObjectName("loVertical")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.lblInforAboutExperiments = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lblInforAboutExperiments.setFont(font)
        self.lblInforAboutExperiments.setObjectName("lblInforAboutExperiments")
        self.verticalLayout.addWidget(self.lblInforAboutExperiments)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.lblFieldsThatContains = QtWidgets.QLabel(self.layoutWidget)
        self.lblFieldsThatContains.setObjectName("lblFieldsThatContains")
        self.horizontalLayout_4.addWidget(self.lblFieldsThatContains)
        self.edtFieldsThatContains = QtWidgets.QLineEdit(self.layoutWidget)
        self.edtFieldsThatContains.setObjectName("edtFieldsThatContains")
        self.horizontalLayout_4.addWidget(self.edtFieldsThatContains)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.tblSummary = QtWidgets.QTableWidget(self.layoutWidget)
        self.tblSummary.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tblSummary.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.tblSummary.setObjectName("tblSummary")
        self.tblSummary.setColumnCount(2)
        self.tblSummary.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tblSummary.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tblSummary.setHorizontalHeaderItem(1, item)
        self.tblSummary.horizontalHeader().setStretchLastSection(True)
        self.verticalLayout.addWidget(self.tblSummary)
        self.loVertical.addLayout(self.verticalLayout)
        self.layoutWidget1 = QtWidgets.QWidget(self.splitter)
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.lblExperiments = QtWidgets.QLabel(self.layoutWidget1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lblExperiments.setFont(font)
        self.lblExperiments.setObjectName("lblExperiments")
        self.verticalLayout_3.addWidget(self.lblExperiments)
        self.tblExperiments = QtWidgets.QTableWidget(self.layoutWidget1)
        self.tblExperiments.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tblExperiments.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tblExperiments.setObjectName("tblExperiments")
        self.tblExperiments.setColumnCount(3)
        self.tblExperiments.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tblExperiments.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tblExperiments.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tblExperiments.setHorizontalHeaderItem(2, item)
        self.tblExperiments.horizontalHeader().setStretchLastSection(True)
        self.verticalLayout_3.addWidget(self.tblExperiments)
        self.btnStart = QtWidgets.QPushButton(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnStart.sizePolicy().hasHeightForWidth())
        self.btnStart.setSizePolicy(sizePolicy)
        self.btnStart.setMinimumSize(QtCore.QSize(0, 40))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/root/edit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnStart.setIcon(icon)
        self.btnStart.setObjectName("btnStart")
        self.verticalLayout_3.addWidget(self.btnStart)
        self.btnRefreshInformationAboutExp = QtWidgets.QPushButton(self.layoutWidget1)
        self.btnRefreshInformationAboutExp.setMinimumSize(QtCore.QSize(0, 40))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/root/refresh.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnRefreshInformationAboutExp.setIcon(icon1)
        self.btnRefreshInformationAboutExp.setObjectName("btnRefreshInformationAboutExp")
        self.verticalLayout_3.addWidget(self.btnRefreshInformationAboutExp)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btnCancelAll = QtWidgets.QPushButton(self.layoutWidget1)
        self.btnCancelAll.setMinimumSize(QtCore.QSize(0, 40))
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/root/delete.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnCancelAll.setIcon(icon2)
        self.btnCancelAll.setObjectName("btnCancelAll")
        self.horizontalLayout.addWidget(self.btnCancelAll)
        self.btnCancel = QtWidgets.QPushButton(self.layoutWidget1)
        self.btnCancel.setMinimumSize(QtCore.QSize(0, 40))
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/root/delete_orange.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnCancel.setIcon(icon3)
        self.btnCancel.setObjectName("btnCancel")
        self.horizontalLayout.addWidget(self.btnCancel)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.btnRemoveExperiments = QtWidgets.QPushButton(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnRemoveExperiments.sizePolicy().hasHeightForWidth())
        self.btnRemoveExperiments.setSizePolicy(sizePolicy)
        self.btnRemoveExperiments.setMinimumSize(QtCore.QSize(0, 40))
        self.btnRemoveExperiments.setStyleSheet("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/root/remove.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnRemoveExperiments.setIcon(icon4)
        self.btnRemoveExperiments.setObjectName("btnRemoveExperiments")
        self.horizontalLayout_2.addWidget(self.btnRemoveExperiments)
        self.btnRemoveSelectedExperiments = QtWidgets.QPushButton(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnRemoveSelectedExperiments.sizePolicy().hasHeightForWidth())
        self.btnRemoveSelectedExperiments.setSizePolicy(sizePolicy)
        self.btnRemoveSelectedExperiments.setMinimumSize(QtCore.QSize(0, 40))
        self.btnRemoveSelectedExperiments.setStyleSheet("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/root/remove_selected.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnRemoveSelectedExperiments.setIcon(icon5)
        self.btnRemoveSelectedExperiments.setObjectName("btnRemoveSelectedExperiments")
        self.horizontalLayout_2.addWidget(self.btnRemoveSelectedExperiments)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.btnClean = QtWidgets.QPushButton(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnClean.sizePolicy().hasHeightForWidth())
        self.btnClean.setSizePolicy(sizePolicy)
        self.btnClean.setMinimumSize(QtCore.QSize(0, 40))
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/root/gnome_clear.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnClean.setIcon(icon6)
        self.btnClean.setObjectName("btnClean")
        self.verticalLayout_2.addWidget(self.btnClean)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.horizontalLayout_3.addWidget(self.splitter)

        self.retranslateUi(SimulationWidget)
        QtCore.QMetaObject.connectSlotsByName(SimulationWidget)

    def retranslateUi(self, SimulationWidget):
        _translate = QtCore.QCoreApplication.translate
        SimulationWidget.setWindowTitle(_translate("SimulationWidget", "Form"))
        self.lblInforAboutExperiments.setText(_translate("SimulationWidget", "Information about experiments"))
        self.lblFieldsThatContains.setText(_translate("SimulationWidget", "Name or value of property contains"))
        self.edtFieldsThatContains.setToolTip(_translate("SimulationWidget", "substring(s) separated by \"|\", that name or value should contain to be included in the output"))
        item = self.tblSummary.horizontalHeaderItem(0)
        item.setText(_translate("SimulationWidget", "Property"))
        item = self.tblSummary.horizontalHeaderItem(1)
        item.setText(_translate("SimulationWidget", "Value"))
        self.lblExperiments.setText(_translate("SimulationWidget", "Experiments"))
        item = self.tblExperiments.horizontalHeaderItem(0)
        item.setText(_translate("SimulationWidget", "Job"))
        item = self.tblExperiments.horizontalHeaderItem(1)
        item.setText(_translate("SimulationWidget", "Device"))
        item = self.tblExperiments.horizontalHeaderItem(2)
        item.setText(_translate("SimulationWidget", "Progress"))
        self.btnStart.setToolTip(_translate("SimulationWidget", "Press F5 for launch simulation"))
        self.btnStart.setText(_translate("SimulationWidget", "Launch simulation"))
        self.btnRefreshInformationAboutExp.setToolTip(_translate("SimulationWidget", "Press F9 for refresh information about experiments"))
        self.btnRefreshInformationAboutExp.setText(_translate("SimulationWidget", " Refresh information about experiments"))
        self.btnRefreshInformationAboutExp.setShortcut(_translate("SimulationWidget", "F9"))
        self.btnCancelAll.setText(_translate("SimulationWidget", " Stop all experiments"))
        self.btnCancel.setToolTip(_translate("SimulationWidget", "Stop after current round selected experiments"))
        self.btnCancel.setText(_translate("SimulationWidget", " Stop selected experiments"))
        self.btnRemoveExperiments.setText(_translate("SimulationWidget", " Remove all experiments"))
        self.btnRemoveSelectedExperiments.setText(_translate("SimulationWidget", "Remove selected experiments"))
        self.btnClean.setText(_translate("SimulationWidget", "Clean checkpoint and log folders"))
import resources_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    SimulationWidget = QtWidgets.QWidget()
    ui = Ui_SimulationWidget()
    ui.setupUi(SimulationWidget)
    SimulationWidget.show()
    sys.exit(app.exec_())
