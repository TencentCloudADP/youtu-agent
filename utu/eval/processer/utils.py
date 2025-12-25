from collections import defaultdict

from ..data import EvaluationSample


class MetricsUtils:
    @staticmethod
    def calculate_overall_metrics(samples: list[EvaluationSample]) -> dict:
        """calculate overall metrics"""
        invalid_count = 0
        for item in samples:
            if item.judged_response == "invalid":
                invalid_count += 1
        total = len(samples)
        correct_count = sum(item.correct for item in samples)
        incorrect_count = total - correct_count - invalid_count
        # confidence_scores = [item.confidence for item in samples if item.judged_response != "invalid"]
        return {
            "Accuracy (%)": round(correct_count / total * 100, 2),
            # "Average Confidence (%)": round(sum(confidence_scores) / total, 2),
            "Details": {
                "correct": correct_count,
                "wrong": incorrect_count,
                "unknown": invalid_count,
                "total": total,
            },
        }

    @staticmethod
    def calculate_level_metrics(samples: list[EvaluationSample]) -> dict:
        """calculate level metrics"""
        level_bin = {}
        for item in samples:
            level = item.level
            if level not in level_bin:
                level_bin[level] = {"correct": 0, "wrong": 0, "unknown": 0}
            if item.judged_response == "invalid":
                level_bin[level]["unknown"] += 1
                continue
            if item.correct:
                level_bin[level]["correct"] += 1
            else:
                level_bin[level]["wrong"] += 1
        for _, counts in level_bin.items():
            total = counts["correct"] + counts["wrong"]
            if total > 0:
                counts["accuracy"] = round(counts["correct"] / total * 100, 4)
            else:
                counts["accuracy"] = 0.0
        return {
            "level_metrics": level_bin,
        }

    @staticmethod
    def calculate_pass_at_k_metrics(samples: list[EvaluationSample], k: int = 1) -> dict:
        """Calculate pass@k metrics by grouping samples by raw_question

        Args:
            samples: List of evaluation samples
            k: Number of attempts to consider.
        """
        # Group samples by raw_question (problem)
        problem_to_scores = defaultdict(list)
        for sample in samples:
            # Skip invalid responses
            if sample.judged_response == "invalid":
                continue
            # Convert correct boolean to score (1 for correct, 0 for incorrect)
            score = 1 if sample.correct else 0
            problem_to_scores[sample.raw_question].append(score)

        # Use the common helper function
        result = MetricsUtils._calculate_pass_at_k_for_problems(problem_to_scores, k)
        return result

    @staticmethod
    def calculate_level_pass_at_k_metrics(samples: list[EvaluationSample], k: int = 1) -> dict:
        """Calculate pass@k metrics by level, grouping samples by level and then by raw_question

        Args:
            samples: List of evaluation samples
            k: Number of attempts to consider. If None, uses max attempts available for each level
        """
        # Group samples by level, then by raw_question (problem)
        level_to_problems = defaultdict(lambda: defaultdict(list))
        for sample in samples:
            # Skip invalid responses
            if sample.judged_response == "invalid":
                continue
            level = sample.level
            # Convert correct boolean to score (1 for correct, 0 for incorrect)
            score = 1 if sample.correct else 0
            level_to_problems[level][sample.raw_question].append(score)

        level_metrics = {}
        for level, problem_to_scores in level_to_problems.items():
            # Use the common helper function
            result = MetricsUtils._calculate_pass_at_k_for_problems(problem_to_scores, k)
            level_metrics[level] = result

        return {
            "level_pass_at_k_metrics": level_metrics,
        }

    @staticmethod
    def calculate_calibration(
        samples: list[EvaluationSample],
        confidence_bins: list[tuple[int, int]] = None,
    ) -> dict:
        """Calculate calibration statistics"""
        confidence_bins = confidence_bins or [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]
        calibration = [{"samples": 0, "correct": 0, "conf_sum": 0} for _ in confidence_bins]
        for record in samples:
            if record.judged_response == "invalid":
                continue
            confidence = record.confidence or 0
            bin_idx = min(confidence // 20, len(confidence_bins) - 1)
            bin_stats = calibration[bin_idx]
            bin_stats["samples"] += 1
            bin_stats["conf_sum"] += confidence
            if record.get("correct", False):
                bin_stats["correct"] += 1
        calibration_error = round(MetricsUtils._calculate_calibration(calibration, len(samples)), 2)
        return {
            "Calibration Error (%)": calibration_error,
        }

    @staticmethod
    def _calculate_calibration(stats: list[dict], total: int) -> float:
        """calculate calibration statistics"""
        error = 0.0
        for bin_stats in stats:
            samples = bin_stats["samples"]
            if not samples:
                continue
            accuracy = bin_stats["correct"] / samples
            avg_conf = bin_stats["conf_sum"] / samples / 100  # convert to 0-1 decimal
            error += (samples / total) * abs(accuracy - avg_conf)
        return error * 100  # convert to percentage

    @staticmethod
    def _calculate_pass_at_k_for_problems(problem_to_scores: dict, k: int = 1) -> dict:
        """Calculate pass@k metrics for a dictionary of problems to scores

        Args:
            problem_to_scores: Dict mapping problem keys to lists of scores (0 or 1)
            k: Number of attempts to consider. If None, uses max attempts available

        Returns:
            Tuple of (pass_at_k_rate, details_dict)
        """
        if not problem_to_scores:
            return 0.0, {"total_problems": 0, "solved_problems": 0, "total_attempts": 0}

        # Calculate pass@k
        pass_at_k_count = 0
        for scores in problem_to_scores.values():
            # Check if any of the first k attempts is correct
            if any(scores[:k]):
                pass_at_k_count += 1

        total_problems = len(problem_to_scores)
        pass_at_k = pass_at_k_count / total_problems if total_problems > 0 else 0.0

        # Add details
        solved_problems = sum(1 for scores in problem_to_scores.values() if max(scores) > 0)
        total_attempts = sum(len(scores) for scores in problem_to_scores.values())

        details = {
            "total_problems": total_problems,
            "solved_problems": solved_problems,
            "unsolved_problems": total_problems - solved_problems,
            "total_attempts": total_attempts,
        }
        return {f"Pass@{k} (%)": round(pass_at_k * 100, 2), "Details": details}
