"""Internationalization support for quantum scheduler."""

import os
import json
import logging
from typing import Dict, Optional, Any
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)


class TranslationManager:
    """Manages translations for different languages."""
    
    def __init__(self, default_locale: str = "en"):
        """Initialize translation manager.
        
        Args:
            default_locale: Default locale code (e.g., 'en', 'es', 'fr')
        """
        self.default_locale = default_locale
        self.current_locale = default_locale
        self.translations: Dict[str, Dict[str, str]] = {}
        self._lock = Lock()
        
        # Load translations
        self._load_translations()
    
    def _load_translations(self):
        """Load translation files."""
        translations_dir = Path(__file__).parent / "translations"
        
        if not translations_dir.exists():
            logger.warning(f"Translations directory not found: {translations_dir}")
            return
        
        for locale_file in translations_dir.glob("*.json"):
            locale_code = locale_file.stem
            try:
                with open(locale_file, 'r', encoding='utf-8') as f:
                    self.translations[locale_code] = json.load(f)
                logger.debug(f"Loaded translations for locale: {locale_code}")
            except Exception as e:
                logger.error(f"Failed to load translations for {locale_code}: {e}")
    
    def set_locale(self, locale: str) -> bool:
        """Set the current locale.
        
        Args:
            locale: Locale code to set
            
        Returns:
            True if locale was set successfully, False otherwise
        """
        with self._lock:
            if locale in self.translations or locale == self.default_locale:
                self.current_locale = locale
                logger.info(f"Locale changed to: {locale}")
                return True
            else:
                logger.warning(f"Locale not available: {locale}")
                return False
    
    def get_current_locale(self) -> str:
        """Get current locale."""
        return self.current_locale
    
    def translate(self, key: str, **kwargs) -> str:
        """Translate a message key.
        
        Args:
            key: Translation key
            **kwargs: Variables for string interpolation
            
        Returns:
            Translated message
        """
        with self._lock:
            # Try current locale first
            if self.current_locale in self.translations:
                translations = self.translations[self.current_locale]
                if key in translations:
                    message = translations[key]
                    try:
                        return message.format(**kwargs)
                    except KeyError as e:
                        logger.warning(f"Missing variable {e} for key '{key}'")
                        return message
            
            # Fallback to default locale
            if self.current_locale != self.default_locale and self.default_locale in self.translations:
                translations = self.translations[self.default_locale]
                if key in translations:
                    message = translations[key]
                    try:
                        return message.format(**kwargs)
                    except KeyError:
                        return message
            
            # Return key if no translation found
            logger.debug(f"No translation found for key: {key}")
            return key
    
    def get_available_locales(self) -> list:
        """Get list of available locales."""
        return list(self.translations.keys())
    
    def add_translation(self, locale: str, key: str, value: str):
        """Add or update a translation.
        
        Args:
            locale: Locale code
            key: Translation key
            value: Translation value
        """
        with self._lock:
            if locale not in self.translations:
                self.translations[locale] = {}
            self.translations[locale][key] = value
    
    def load_from_dict(self, locale: str, translations: Dict[str, str]):
        """Load translations from a dictionary.
        
        Args:
            locale: Locale code
            translations: Dictionary of translations
        """
        with self._lock:
            self.translations[locale] = translations.copy()


# Global translation manager instance
_translation_manager = TranslationManager()


def get_translation_manager() -> TranslationManager:
    """Get the global translation manager."""
    return _translation_manager


def set_locale(locale: str) -> bool:
    """Set the global locale."""
    return _translation_manager.set_locale(locale)


def get_locale() -> str:
    """Get the current global locale."""
    return _translation_manager.get_current_locale()


def translate(key: str, **kwargs) -> str:
    """Translate a message key using the global translation manager."""
    return _translation_manager.translate(key, **kwargs)


# Convenience alias
t = translate


# Initialize default translations
_default_translations = {
    "en": {
        "scheduler.initializing": "Initializing quantum scheduler with backend: {backend}",
        "scheduler.solving": "Solving scheduling problem with {num_agents} agents and {num_tasks} tasks",
        "scheduler.solution_found": "Solution found with {num_assignments} assignments and cost {cost:.2f}",
        "scheduler.no_solution": "No feasible solution found",
        "scheduler.timeout": "Solver timeout after {timeout} seconds",
        "validation.empty_id": "ID cannot be empty",
        "validation.invalid_skills": "Invalid skills list",
        "validation.invalid_capacity": "Capacity must be positive",
        "validation.invalid_duration": "Duration must be positive",
        "validation.invalid_priority": "Priority must be positive",
        "error.backend_error": "Backend error: {error}",
        "error.solver_error": "Solver error: {error}",
        "error.validation_error": "Validation error: {error}",
        "health.backend_healthy": "Backend is healthy",
        "health.backend_unhealthy": "Backend is unhealthy: {reason}",
        "health.overall_healthy": "System is healthy",
        "health.overall_degraded": "System performance is degraded",
        "health.overall_critical": "System is in critical state",
        "metrics.success_rate": "Success rate: {rate:.1%}",
        "metrics.avg_response_time": "Average response time: {time:.2f}s",
        "metrics.total_requests": "Total requests: {count}",
        "security.threat_detected": "Security threat detected: {threat_type}",
        "security.rate_limit_exceeded": "Rate limit exceeded for: {identifier}",
        "security.invalid_input": "Invalid input detected and sanitized",
    },
    "es": {
        "scheduler.initializing": "Inicializando planificador cuántico con backend: {backend}",
        "scheduler.solving": "Resolviendo problema de planificación con {num_agents} agentes y {num_tasks} tareas",
        "scheduler.solution_found": "Solución encontrada con {num_assignments} asignaciones y costo {cost:.2f}",
        "scheduler.no_solution": "No se encontró solución factible",
        "scheduler.timeout": "Tiempo de espera del solver agotado después de {timeout} segundos",
        "validation.empty_id": "El ID no puede estar vacío",
        "validation.invalid_skills": "Lista de habilidades inválida",
        "validation.invalid_capacity": "La capacidad debe ser positiva",
        "validation.invalid_duration": "La duración debe ser positiva",
        "validation.invalid_priority": "La prioridad debe ser positiva",
        "error.backend_error": "Error de backend: {error}",
        "error.solver_error": "Error del solver: {error}",
        "error.validation_error": "Error de validación: {error}",
        "health.backend_healthy": "El backend está saludable",
        "health.backend_unhealthy": "El backend no está saludable: {reason}",
        "health.overall_healthy": "El sistema está saludable",
        "health.overall_degraded": "El rendimiento del sistema está degradado",
        "health.overall_critical": "El sistema está en estado crítico",
        "metrics.success_rate": "Tasa de éxito: {rate:.1%}",
        "metrics.avg_response_time": "Tiempo de respuesta promedio: {time:.2f}s",
        "metrics.total_requests": "Total de solicitudes: {count}",
        "security.threat_detected": "Amenaza de seguridad detectada: {threat_type}",
        "security.rate_limit_exceeded": "Límite de tasa excedido para: {identifier}",
        "security.invalid_input": "Entrada inválida detectada y sanitizada",
    },
    "fr": {
        "scheduler.initializing": "Initialisation du planificateur quantique avec backend: {backend}",
        "scheduler.solving": "Résolution du problème de planification avec {num_agents} agents et {num_tasks} tâches",
        "scheduler.solution_found": "Solution trouvée avec {num_assignments} affectations et coût {cost:.2f}",
        "scheduler.no_solution": "Aucune solution réalisable trouvée",
        "scheduler.timeout": "Délai d'attente du solveur dépassé après {timeout} secondes",
        "validation.empty_id": "L'ID ne peut pas être vide",
        "validation.invalid_skills": "Liste de compétences invalide",
        "validation.invalid_capacity": "La capacité doit être positive",
        "validation.invalid_duration": "La durée doit être positive",
        "validation.invalid_priority": "La priorité doit être positive",
        "error.backend_error": "Erreur de backend: {error}",
        "error.solver_error": "Erreur du solveur: {error}",
        "error.validation_error": "Erreur de validation: {error}",
        "health.backend_healthy": "Le backend est en bonne santé",
        "health.backend_unhealthy": "Le backend n'est pas en bonne santé: {reason}",
        "health.overall_healthy": "Le système est en bonne santé",
        "health.overall_degraded": "Les performances du système sont dégradées",
        "health.overall_critical": "Le système est dans un état critique",
        "metrics.success_rate": "Taux de succès: {rate:.1%}",
        "metrics.avg_response_time": "Temps de réponse moyen: {time:.2f}s",
        "metrics.total_requests": "Total des requêtes: {count}",
        "security.threat_detected": "Menace de sécurité détectée: {threat_type}",
        "security.rate_limit_exceeded": "Limite de taux dépassée pour: {identifier}",
        "security.invalid_input": "Entrée invalide détectée et assainie",
    },
    "de": {
        "scheduler.initializing": "Quantenplaner wird mit Backend initialisiert: {backend}",
        "scheduler.solving": "Planungsproblem wird gelöst mit {num_agents} Agenten und {num_tasks} Aufgaben",
        "scheduler.solution_found": "Lösung gefunden mit {num_assignments} Zuweisungen und Kosten {cost:.2f}",
        "scheduler.no_solution": "Keine machbare Lösung gefunden",
        "scheduler.timeout": "Solver-Timeout nach {timeout} Sekunden",
        "validation.empty_id": "ID darf nicht leer sein",
        "validation.invalid_skills": "Ungültige Fertigkeitenliste",
        "validation.invalid_capacity": "Kapazität muss positiv sein",
        "validation.invalid_duration": "Dauer muss positiv sein",
        "validation.invalid_priority": "Priorität muss positiv sein",
        "error.backend_error": "Backend-Fehler: {error}",
        "error.solver_error": "Solver-Fehler: {error}",
        "error.validation_error": "Validierungsfehler: {error}",
        "health.backend_healthy": "Backend ist gesund",
        "health.backend_unhealthy": "Backend ist ungesund: {reason}",
        "health.overall_healthy": "System ist gesund",
        "health.overall_degraded": "Systemleistung ist beeinträchtigt",
        "health.overall_critical": "System ist in kritischem Zustand",
        "metrics.success_rate": "Erfolgsrate: {rate:.1%}",
        "metrics.avg_response_time": "Durchschnittliche Antwortzeit: {time:.2f}s",
        "metrics.total_requests": "Gesamte Anfragen: {count}",
        "security.threat_detected": "Sicherheitsbedrohung erkannt: {threat_type}",
        "security.rate_limit_exceeded": "Rate-Limit überschritten für: {identifier}",
        "security.invalid_input": "Ungültige Eingabe erkannt und bereinigt",
    },
    "ja": {
        "scheduler.initializing": "バックエンド {backend} でクォンタムスケジューラーを初期化中",
        "scheduler.solving": "{num_agents} エージェントと {num_tasks} タスクでスケジューリング問題を解決中",
        "scheduler.solution_found": "{num_assignments} 割り当てとコスト {cost:.2f} で解が見つかりました",
        "scheduler.no_solution": "実行可能な解が見つかりませんでした",
        "scheduler.timeout": "{timeout} 秒後にソルバーがタイムアウトしました",
        "validation.empty_id": "IDは空にできません",
        "validation.invalid_skills": "無効なスキルリスト",
        "validation.invalid_capacity": "容量は正の値である必要があります",
        "validation.invalid_duration": "期間は正の値である必要があります",
        "validation.invalid_priority": "優先度は正の値である必要があります",
        "error.backend_error": "バックエンドエラー: {error}",
        "error.solver_error": "ソルバーエラー: {error}",
        "error.validation_error": "検証エラー: {error}",
        "health.backend_healthy": "バックエンドは健全です",
        "health.backend_unhealthy": "バックエンドは不健全です: {reason}",
        "health.overall_healthy": "システムは健全です",
        "health.overall_degraded": "システムパフォーマンスが低下しています",
        "health.overall_critical": "システムが危険な状態です",
        "metrics.success_rate": "成功率: {rate:.1%}",
        "metrics.avg_response_time": "平均応答時間: {time:.2f}秒",
        "metrics.total_requests": "総リクエスト数: {count}",
        "security.threat_detected": "セキュリティ脅威を検出: {threat_type}",
        "security.rate_limit_exceeded": "レート制限を超過: {identifier}",
        "security.invalid_input": "無効な入力を検出してサニタイズしました",
    },
    "zh": {
        "scheduler.initializing": "正在使用后端 {backend} 初始化量子调度器",
        "scheduler.solving": "正在解决包含 {num_agents} 个代理和 {num_tasks} 个任务的调度问题",
        "scheduler.solution_found": "找到解决方案：{num_assignments} 个分配，成本 {cost:.2f}",
        "scheduler.no_solution": "未找到可行解决方案",
        "scheduler.timeout": "求解器在 {timeout} 秒后超时",
        "validation.empty_id": "ID不能为空",
        "validation.invalid_skills": "无效的技能列表",
        "validation.invalid_capacity": "容量必须为正数",
        "validation.invalid_duration": "持续时间必须为正数",
        "validation.invalid_priority": "优先级必须为正数",
        "error.backend_error": "后端错误: {error}",
        "error.solver_error": "求解器错误: {error}",
        "error.validation_error": "验证错误: {error}",
        "health.backend_healthy": "后端状态良好",
        "health.backend_unhealthy": "后端状态不良: {reason}",
        "health.overall_healthy": "系统状态良好",
        "health.overall_degraded": "系统性能下降",
        "health.overall_critical": "系统处于危险状态",
        "metrics.success_rate": "成功率: {rate:.1%}",
        "metrics.avg_response_time": "平均响应时间: {time:.2f}秒",
        "metrics.total_requests": "总请求数: {count}",
        "security.threat_detected": "检测到安全威胁: {threat_type}",
        "security.rate_limit_exceeded": "超出速率限制: {identifier}",
        "security.invalid_input": "检测到无效输入并已清理",
    }
}

# Load default translations
for locale, translations in _default_translations.items():
    _translation_manager.load_from_dict(locale, translations)

logger.info(f"Loaded translations for {len(_default_translations)} locales")


__all__ = [
    "TranslationManager",
    "get_translation_manager", 
    "set_locale",
    "get_locale",
    "translate",
    "t"
]