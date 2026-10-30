import os
import csv
import json
import numpy as np

class BlindEvaluator:
    """
    Blind Evaluation Engine.
    Stores ground truth annotations in a sealed vault during testing.
    Unseals and computes statistical accuracy metrics (MAE, RMSE, Pass Rate)
    only after all test runs have concluded.
    """
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        self.sealed_vault = {}
        self.is_unsealed = False
        os.makedirs(self.output_dir, exist_ok=True)

    def seal_ground_truth(self, video_id, true_speed_kmh, cpa_time_sec):
        """
        Stores ground-truth annotations inside the sealed vault.
        """
        self.sealed_vault[video_id] = {
            "true_speed_kmh": float(true_speed_kmh),
            "cpa_time_sec": float(cpa_time_sec)
        }

    def unseal_and_evaluate(self, predictions):
        """
        Unseals the ground truth and performs statistical audit against model predictions.
        
        Args:
            predictions: dict of {video_id: entry_dict_from_memory}
            
        Returns:
            dict with detailed evaluation table, MAE, RMSE, Pass Rate, and CSV path.
        """
        self.is_unsealed = True
        eval_rows = []
        
        errors_vision = []
        errors_audio = []
        errors_fused = []

        pass_3kmh = 0
        pass_5kmh = 0
        total = 0

        csv_path = os.path.join(self.output_dir, "slobodan_vs13_accuracy_evaluation.csv")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Video_ID",
                "Vehicle_Model",
                "Ground_Truth_Speed_KMH",
                "Vision_Speed_KMH",
                "Vision_Error_KMH",
                "Audio_Speed_KMH",
                "Audio_Error_KMH",
                "Fused_Speed_KMH",
                "Fused_Error_KMH",
                "Accuracy_Pct",
                "Pass_Within_5KMH"
            ])

            for vid, pred in predictions.items():
                if vid not in self.sealed_vault:
                    continue

                total += 1
                gt = self.sealed_vault[vid]["true_speed_kmh"]
                v_vis = pred["visual"]["speed_kmh"]
                v_aud = pred["acoustic"]["speed_kmh"]
                v_fus = pred["fused_speed_kmh"]

                err_vis = abs(v_vis - gt)
                err_aud = abs(v_aud - gt)
                err_fus = abs(v_fus - gt)

                errors_vision.append(err_vis)
                errors_audio.append(err_aud)
                errors_fused.append(err_fus)

                accuracy_pct = max(0.0, 100.0 - (err_fus / gt * 100.0))
                is_pass_5 = (err_fus <= 5.0)
                is_pass_3 = (err_fus <= 3.0)

                if is_pass_5:
                    pass_5kmh += 1
                if is_pass_3:
                    pass_3kmh += 1

                row = {
                    "video_id": vid,
                    "model": pred.get("vehicle_model", "Unknown"),
                    "true_speed": gt,
                    "vision_speed": v_vis,
                    "vision_error": round(err_vis, 2),
                    "audio_speed": v_aud,
                    "audio_error": round(err_aud, 2),
                    "fused_speed": v_fus,
                    "fused_error": round(err_fus, 2),
                    "accuracy_pct": round(accuracy_pct, 1),
                    "passed_5kmh": is_pass_5
                }
                eval_rows.append(row)

                writer.writerow([
                    vid,
                    row["model"],
                    gt,
                    v_vis,
                    row["vision_error"],
                    v_aud,
                    row["audio_error"],
                    v_fus,
                    row["fused_error"],
                    f"{accuracy_pct:.1f}%",
                    "PASS" if is_pass_5 else "FAIL"
                ])

        if total == 0:
            return {"error": "No matching predictions found in sealed vault."}

        mae_vis = float(np.mean(errors_vision))
        mae_aud = float(np.mean(errors_audio))
        mae_fus = float(np.mean(errors_fused))

        rmse_fus = float(np.sqrt(np.mean(np.array(errors_fused) ** 2)))
        mean_acc = float(np.mean([r["accuracy_pct"] for r in eval_rows]))

        summary = {
            "total_evaluated": total,
            "fused_mae_kmh": round(mae_fus, 2),
            "vision_mae_kmh": round(mae_vis, 2),
            "audio_mae_kmh": round(mae_aud, 2),
            "fused_rmse_kmh": round(rmse_fus, 2),
            "mean_accuracy_pct": round(mean_acc, 2),
            "pass_rate_5kmh_pct": round((pass_5kmh / total) * 100.0, 1),
            "pass_rate_3kmh_pct": round((pass_3kmh / total) * 100.0, 1),
            "csv_report": os.path.abspath(csv_path),
            "detailed_rows": eval_rows
        }

        with open(os.path.join(self.output_dir, "slobodan_vs13_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
