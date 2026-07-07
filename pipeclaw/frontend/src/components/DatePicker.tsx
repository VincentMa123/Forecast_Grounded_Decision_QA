import React from 'react';
import './DatePicker.css';

interface DatePickerProps {
  selectedDate: string;
  availableDates: string[];
  onChange: (date: string) => void;
  label?: string;
}

export const DatePicker: React.FC<DatePickerProps> = ({
  selectedDate,
  availableDates,
  onChange,
  label = '选择日期'
}) => {
  return (
    <div className="date-picker">
      <label htmlFor="date-select">{label}:</label>
      <select
        id="date-select"
        value={selectedDate}
        onChange={(e) => onChange(e.target.value)}
        className="date-select"
      >
        {availableDates.length === 0 ? (
          <option value="">暂无可用日期</option>
        ) : (
          availableDates.map((date) => (
            <option key={date} value={date}>
              {date}
            </option>
          ))
        )}
      </select>
    </div>
  );
};
