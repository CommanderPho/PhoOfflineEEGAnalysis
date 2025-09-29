#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
from copy import deepcopy

from enum import Enum
from datetime import datetime, timezone
from attrs import define, field, Factory

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QWidget, QMessageBox, QDataWidgetMapper, QToolTip, QLabel, QTimeEdit, QCalendarWidget
from PyQt5.QtGui import QCursor, QColor, QPalette
from PyQt5.QtCore import QDir, QTime, QDate, QDateTime, QTimer, Qt, QModelIndex, QSortFilterProxyModel, QObject, QEvent, pyqtSignal, pyqtSlot, QMargins, QRect, QRectF

dt_col_names = ['recording_datetime', 'recording_day_date']
timestamp_column_names = ['created_at', 'first_timestamp', 'last_timestamp']
timestamp_dt_column_names = ['created_at_dt', 'first_timestamp_dt', 'last_timestamp_dt']

# from dose_analysis_python.FileImportExport.NoteImporter import NightBingeStatus
@define(slots=False)
class CalendarDatasource:
    """ from PhoLabStreamingReceiver.UI.CustomCalendarWidget import CalendarDatasource
    """
    xdf_stream_infos_df: pd.DataFrame = field()
    eeg_only_stream_infos_df: pd.DataFrame = field(init=False)

    def __attrs_post_init__(self):
        self.xdf_stream_infos_df[dt_col_names] = self.xdf_stream_infos_df[dt_col_names].convert_dtypes()
        self.eeg_only_stream_infos_df = deepcopy(self.xdf_stream_infos_df.reset_index(drop=False, inplace=False, names='idx_bak'))
        self.eeg_only_stream_infos_df = self.eeg_only_stream_infos_df[self.eeg_only_stream_infos_df['name'] == "Epoc X"]
        for a_ts_col_name, a_ts_dt_col_name in zip(timestamp_column_names, timestamp_dt_column_names):
            if a_ts_dt_col_name not in self.eeg_only_stream_infos_df.columns:
                try:
                    # a_ts_dt_col_name: str = f'{a_ts_col_name}_dt'
                    # xdf_stream_infos_df[a_ts_dt_col_name] = xdf_stream_infos_df['recording_datetime'] + [pd.Timestamp(v) for v in xdf_stream_infos_df[a_ts_col_name].to_numpy()]
                    # xdf_stream_infos_df[a_ts_dt_col_name] = xdf_stream_infos_df['recording_datetime'] + [pd.Timedelta(seconds=float(v)) for v in xdf_stream_infos_df[a_ts_col_name].to_numpy()]
                    # xdf_stream_infos_df[a_ts_dt_col_name] = [pd.Timedelta(seconds=float(v)) for v in xdf_stream_infos_df[a_ts_col_name].to_numpy()]
                    self.eeg_only_stream_infos_df[a_ts_dt_col_name] = [pd.Timedelta(seconds=float(v)) for v in self.eeg_only_stream_infos_df[a_ts_col_name].to_numpy()]
                    self.eeg_only_stream_infos_df[a_ts_dt_col_name] = self.eeg_only_stream_infos_df['recording_datetime'] + self.eeg_only_stream_infos_df[a_ts_dt_col_name]
                    
                except (ValueError, AttributeError) as e:
                    print(f'failed to add column "{a_ts_dt_col_name}" due to error: {e}. Skipping col.')
                except Exception as e:
                    raise e
            

    # @property
    # def eeg_only_stream_infos_df(self):
    #     """The eeg_only_stream_infos_df property."""
    #     eeg_only_df = deepcopy(self.xdf_stream_infos_df.reset_index(drop=False, inplace=False))
    #     return eeg_only_df[eeg_only_df['name'] == "Epoc X"]
        
    @property
    def records_per_day_date(self):
        """The records_per_day_date property."""
        return self.eeg_only_stream_infos_df['recording_day_date'].value_counts().to_dict()

        
    def get_records_for_day(self, day_date: datetime):
        day_date = day_date.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(tz=timezone.utc)
        # return self.xdf_stream_infos_df[self.xdf_stream_infos_df['recording_day_date'] == day_date]  
        return self.eeg_only_stream_infos_df[self.eeg_only_stream_infos_df['recording_day_date'] == day_date]  


class CustomDataDisplayingCalendar(QCalendarWidget):
    def __init__(self, datasource=None, parent=None):
        super().__init__(parent=parent)
        self._datasource = datasource
        self.color = QColor(self.palette().color(QPalette.Highlight))
        self.color.setAlpha(64)

        self.note_missing_color = QColor(30, 30, 30, 40)
        self.unlabled_color = QColor(0, 10, 200, 10)

        self.binge_color = QColor(200, 0, 0, 64)
        # self.binge_color.setAlpha(64)

        self.good_color = QColor(0,200,0, 64)
        # self.good_color.setAlpha(64)


        self.selectionChanged.connect(self.updateCells)

    @property
    def datasource(self) -> CalendarDatasource:
        """The datesource property."""
        return self._datasource
    @datasource.setter
    def datasource(self, value: CalendarDatasource):
        self._datasource = value

    def set_datasource(self, data_source):
        self._datasource = data_source
        self.updateCells()

    def paintCell(self, painter, rect, date):
        # date: QDate object
        QCalendarWidget.paintCell(self, painter, rect, date)
        
        if self.datasource is not None:
            # Got noteManager, shade the cell based on the binge status:
            ## find the note with this date:
            converted_date = QDateTime(date).toPyDateTime()
            converted_date = converted_date.replace(tzinfo=timezone.utc)

            # converted_date = QDate(date).toString()
            # converted_date = date.toString()
            found_df: pd.DataFrame = self.datasource.get_records_for_day(converted_date)
            # print(self.noteManager)
            # print(f'found_df: {found_df}')
            
            if (found_df is None) or len(found_df) == 0:
                # # print('WARNING: calendar could not find note for date {}. Skipping.'.format(converted_date))
                # desired_margin_inset = rect.width() * 0.15
                # # desired_text_rect = rect.marginsAdded(QMargins(desired_margin_inset, 0, desired_margin_inset, 0))
                # # rect.setWidth(desired_margin_inset)
                # rect.setHeight(desired_margin_inset)
                # # rect.setLeft(desired_margin_inset)

                # painter.fillRect(rect, self.note_missing_color)
                # # painter.drawText(desired_text_rect.topLeft(), '!') # Draw an indicator to show that the note is missing.
                pass
            else:
                # Get width of the rect to draw: Limit it to about 10% of the cell width, affixed to the left side (where the morning would be in a timeline).
                # desired_width = rect.width() * 0.10
                desired_width = rect.width() * 0.90
                # rect.setWidth(desired_width)
                
                print(f'converted_date: {converted_date}, rect: {rect}, len(found_df): {len(found_df)}')
                # found_index = found_df[0]
                # found_obj = found_df[1]                
                # found_df['first_timestamp_rel']
                for a_row in found_df.itertuples():                
                    # a_rect = deepcopy(rect)
                                    
                    desired_height: float = float(rect.height()) * float(a_row.duration_rel)
                    # desired_top: float = float(a_rect.top())
                    desired_start_y: float = float(rect.height()) * float(a_row.first_timestamp_rel)
                    desired_top: float = float(rect.top()) + desired_start_y
                    a_rect = QRectF(rect.left(), desired_top, desired_width, desired_height)    
                    # a_rect.setTop(desired_top)
                    # a_rect.setHeight(desired_height)
                    # a_rect.setWidth(desired_width)
                    print(f'\t\ta_rect: {a_rect}')
                    # painter.fillRect(a_rect, self.good_color)
                    painter.fillRect(a_rect, self.binge_color)
                    

                # # get note object's properties:
                # curr_binge_status = found_obj.userData['binge_status']
                # # print('Found matching note for calendar date {}! found_tuple: {}, curr_binge_status: {}'.format(converted_date, found_tuple, curr_binge_status))

                # # Get width of the rect to draw: Limit it to about 10% of the cell width, affixed to the left side (where the morning would be in a timeline).
                # desired_width = rect.width() * 0.10
                # rect.setWidth(desired_width)

                # if curr_binge_status is NightBingeStatus.Binge:
                #     painter.fillRect(rect, self.binge_color)
                #     # painter.fillRect(rect, self.color)
                #     pass

                # elif curr_binge_status is NightBingeStatus.NonBinge:
                #     painter.fillRect(rect, self.good_color)
                #     pass
                # else:
                #     painter.fillRect(rect, self.unlabled_color)
                #     pass

            # first_day = self.firstDayOfWeek()
            # last_day = first_day + 6
            # current_date = self.selectedDate()
            # current_day = current_date.dayOfWeek()

            # if first_day <= current_day:
            #     first_date = current_date.addDays(first_day - current_day)
            # else:
            #     first_date = current_date.addDays(first_day - 7 - current_day)
            # last_date = first_date.addDays(6)

            # if first_date <= date <= last_date:
            #     painter.fillRect(rect, self.color)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CustomDataDisplayingCalendar()
    ex.show()
    sys.exit(app.exec_())
