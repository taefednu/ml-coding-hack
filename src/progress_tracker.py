"""
Класс для отслеживания прогресса обучения с процентами и временем.
"""

import time
from datetime import datetime, timedelta


class ProgressTracker:
    """Отслеживает прогресс обучения с процентами и временем."""
    
    def __init__(self, total_steps: int, step_name: str = "Шаг"):
        self.total_steps = total_steps
        self.current_step = 0
        self.step_name = step_name
        self.start_time = time.time()
        self.step_times = []
    
    def update(self, step: int = None, message: str = ""):
        """Обновляет прогресс."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        elapsed = time.time() - self.start_time
        
        if self.current_step > 0:
            avg_time_per_step = elapsed / self.current_step
            remaining_steps = self.total_steps - self.current_step
            estimated_remaining = avg_time_per_step * remaining_steps
        else:
            estimated_remaining = 0
        
        percentage = (self.current_step / self.total_steps) * 100
        
        # Форматирование времени
        elapsed_str = self._format_time(elapsed)
        remaining_str = self._format_time(estimated_remaining)
        
        # Прогресс-бар
        bar_length = 40
        filled = int(bar_length * self.current_step / self.total_steps)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        # Вывод
        print(f"\r   [{bar}] {percentage:5.1f}% | {self.current_step}/{self.total_steps} | "
              f"⏱ {elapsed_str} | ⏳ Осталось: {remaining_str} {message}", end="", flush=True)
        
        if self.current_step >= self.total_steps:
            print()  # Новая строка в конце
    
    def _format_time(self, seconds: float) -> str:
        """Форматирует время в читаемый вид."""
        if seconds < 60:
            return f"{int(seconds)}с"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}м {secs}с"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}ч {minutes}м"
    
    def finish(self):
        """Завершает отслеживание."""
        total_time = time.time() - self.start_time
        print(f"\n   ✅ Завершено за: {self._format_time(total_time)}")
        return total_time

