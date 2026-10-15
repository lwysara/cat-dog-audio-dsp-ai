            if (
                medians
                and x is not None
                and confidence is not None
                and hasattr(model, "predict_proba")
                and all(name in medians for name in feature_names)
            ):
                pred_class_index = classes.index(pred)
                impacts = []

                for name in feature_names:
                    altered = x.copy()
                    altered.loc[0, name] = float(medians[name])
                    altered_scaled = scaler.transform(altered)
                    altered_probs = model.predict_proba(altered_scaled)[0]
                    altered_prob = float(altered_probs[pred_class_index])

                    impacts.append({
                        "Feature": name,
                        "Original_Value": float(x.loc[0, name]),
                        "Training_Median": float(medians[name]),
                        "Impact": float(confidence - altered_prob),
                    })

                local = pd.DataFrame(impacts)
                local["Abs_Impact"] = local["Impact"].abs()
                local = local.sort_values(
                    "Abs_Impact",
                    ascending=False,
                )

                chart = local.head(10).sort_values("Impact")
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.barh(chart["Feature"], chart["Impact"])
                ax.axvline(0, linestyle="--")
                ax.set_xlabel(
                    "Original class probability − probability after median replacement"
                )
                ax.grid(axis="x", alpha=0.2)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                shown = local.head(15).copy()
                shown["Impact (%)"] = (shown["Impact"] * 100).round(2)
                st.dataframe(
                    shown[
                        [
                            "Feature",
                            "Original_Value",
                            "Training_Median",
                            "Impact (%)",
                        ]
                    ],
                    use_container_width=True,
                )
            else:
                st.info(
                    "模型檔未提供完整 training_medians，或模型不支援機率輸出；"
                    "因此略過 Local Feature Impact。"
                )

            st.subheader("⑧ 模型評估（訓練時保存）")
            if "test_accuracy" in bundle:
                st.metric(
                    "Hold-out Test Accuracy",
                    f"{float(bundle['test_accuracy']) * 100:.2f}%",
                )
                st.caption(
                    "這是訓練時保留測試集的結果，不是目前上傳音訊的正確率。"
                )

            if "confusion_matrix" in bundle:
                cm = np.asarray(bundle["confusion_matrix"])
                if cm.ndim == 2:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    im = ax.imshow(cm)
                    tick_labels = ["Cat", "Dog"]
                    ax.set_xticks(range(cm.shape[1]))
                    ax.set_yticks(range(cm.shape[0]))
                    ax.set_xticklabels(tick_labels[:cm.shape[1]])
                    ax.set_yticklabels(tick_labels[:cm.shape[0]])
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title("Confusion Matrix")
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax.text(
                                j,
                                i,
                                int(cm[i, j]),
                                ha="center",
                                va="center",
                            )
                    fig.colorbar(im, ax=ax)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            if "classification_report" in bundle:
                with st.expander("查看 Classification Report"):
                    report_df = pd.DataFrame(bundle["classification_report"]).T
                    st.dataframe(report_df, use_container_width=True)

        st.subheader("⑨ Engineering Interpretation")
        st.markdown(
            """
```text
Audio Input
    ↓
Audio Decode / Mono / 22.05 kHz
    ↓
300–4000 Hz Band-pass Filter
    ↓
22 DSP Features
    ↓
Cat / Dog Random Forest (classification mode only)
    ↓
Prediction + DSP Plots + Explainability
```

**注意：Feature Importance / Local Feature Impact 是模型解釋，不是物理因果證明。**
"""
        )

    except Exception as exc:
        st.error(f"音訊分析失敗：{exc}")
        st.exception(exc)
