#!/usr/bin/env python3
"""
Model Comparison and Selection Example

This example demonstrates the new model comparison and selection utilities
added to the ModelEvaluator class. It shows how to:

1. Create comparison tables for multiple models
2. Perform statistical significance testing
3. Select the best model based on business metrics
4. Generate comprehensive selection reports
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from models.model_evaluator import ModelEvaluator


def create_sample_evaluation_results():
    """Create sample evaluation results for demonstration."""
    
    # Simulate evaluation results from 3 different fraud detection models
    model_results = [
        {
            'model_name': 'LogisticRegression',
            'accuracy': 0.856,
            'precision': 0.742,
            'recall': 0.798,
            'f1_score': 0.769,
            'auc_roc': 0.883,
            'specificity': 0.871,
            'false_positive_rate': 0.129,
            'false_negative_rate': 0.202,
            'test_samples': 1000,
            'positive_samples': 200,
            'negative_samples': 800,
            'true_positives': 160,
            'true_negatives': 696,
            'false_positives': 104,
            'false_negatives': 40
        },
        {
            'model_name': 'RandomForest',
            'accuracy': 0.884,
            'precision': 0.816,
            'recall': 0.778,
            'f1_score': 0.797,
            'auc_roc': 0.912,
            'specificity': 0.910,
            'false_positive_rate': 0.090,
            'false_negative_rate': 0.222,
            'test_samples': 1000,
            'positive_samples': 200,
            'negative_samples': 800,
            'true_positives': 156,
            'true_negatives': 728,
            'false_positives': 72,
            'false_negatives': 44
        },
        {
            'model_name': 'XGBoost',
            'accuracy': 0.900,
            'precision': 0.848,
            'recall': 0.820,
            'f1_score': 0.834,
            'auc_roc': 0.934,
            'specificity': 0.920,
            'false_positive_rate': 0.080,
            'false_negative_rate': 0.180,
            'test_samples': 1000,
            'positive_samples': 200,
            'negative_samples': 800,
            'true_positives': 164,
            'true_negatives': 736,
            'false_positives': 64,
            'false_negatives': 36
        }
    ]
    
    return model_results


def create_sample_cv_results():
    """Create sample cross-validation results for significance testing."""
    
    cv_results = [
        {
            'cv_summary': {
                'f1_score_mean': 0.769,
                'f1_score_std': 0.028,
                'f1_score_scores': [0.745, 0.782, 0.761, 0.789, 0.768],
                'precision_mean': 0.742,
                'precision_std': 0.035,
                'precision_scores': [0.718, 0.765, 0.738, 0.771, 0.748],
                'recall_mean': 0.798,
                'recall_std': 0.022,
                'recall_scores': [0.785, 0.812, 0.795, 0.821, 0.797]
            },
            'fold_details': [
                {'fold': 1, 'metrics': {'model_name': 'LogisticRegression'}},
                {'fold': 2, 'metrics': {'model_name': 'LogisticRegression'}},
                {'fold': 3, 'metrics': {'model_name': 'LogisticRegression'}},
                {'fold': 4, 'metrics': {'model_name': 'LogisticRegression'}},
                {'fold': 5, 'metrics': {'model_name': 'LogisticRegression'}}
            ]
        },
        {
            'cv_summary': {
                'f1_score_mean': 0.797,
                'f1_score_std': 0.021,
                'f1_score_scores': [0.789, 0.812, 0.785, 0.821, 0.798],
                'precision_mean': 0.816,
                'precision_std': 0.028,
                'precision_scores': [0.798, 0.834, 0.812, 0.845, 0.791],
                'recall_mean': 0.778,
                'recall_std': 0.025,
                'recall_scores': [0.765, 0.791, 0.772, 0.798, 0.764]
            },
            'fold_details': [
                {'fold': 1, 'metrics': {'model_name': 'RandomForest'}},
                {'fold': 2, 'metrics': {'model_name': 'RandomForest'}},
                {'fold': 3, 'metrics': {'model_name': 'RandomForest'}},
                {'fold': 4, 'metrics': {'model_name': 'RandomForest'}},
                {'fold': 5, 'metrics': {'model_name': 'RandomForest'}}
            ]
        },
        {
            'cv_summary': {
                'f1_score_mean': 0.834,
                'f1_score_std': 0.018,
                'f1_score_scores': [0.821, 0.845, 0.828, 0.851, 0.825],
                'precision_mean': 0.848,
                'precision_std': 0.019,
                'precision_scores': [0.834, 0.862, 0.841, 0.868, 0.835],
                'recall_mean': 0.820,
                'recall_std': 0.020,
                'recall_scores': [0.808, 0.832, 0.815, 0.838, 0.807]
            },
            'fold_details': [
                {'fold': 1, 'metrics': {'model_name': 'XGBoost'}},
                {'fold': 2, 'metrics': {'model_name': 'XGBoost'}},
                {'fold': 3, 'metrics': {'model_name': 'XGBoost'}},
                {'fold': 4, 'metrics': {'model_name': 'XGBoost'}},
                {'fold': 5, 'metrics': {'model_name': 'XGBoost'}}
            ]
        }
    ]
    
    return cv_results


def demonstrate_comparison_table():
    """Demonstrate comparison table creation."""
    print("=" * 60)
    print("1. MODEL COMPARISON TABLE")
    print("=" * 60)
    
    evaluator = ModelEvaluator()
    model_results = create_sample_evaluation_results()
    
    # Create comparison table
    comparison_df = evaluator.create_comparison_table(model_results)
    
    print("Performance Comparison Table:")
    print("-" * 40)
    
    # Display key metrics with formatting
    display_cols = ['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    print(comparison_df[display_cols].round(3).to_string(index=False))
    
    print("\nModel Rankings:")
    print("-" * 20)
    ranking_cols = ['model_name', 'f1_score_rank', 'auc_roc_rank', 'precision_rank']
    print(comparison_df[ranking_cols].to_string(index=False))
    
    return comparison_df


def demonstrate_statistical_significance():
    """Demonstrate statistical significance testing."""
    print("\n" + "=" * 60)
    print("2. STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 60)
    
    evaluator = ModelEvaluator()
    cv_results = create_sample_cv_results()
    
    # Extract F1 scores for pairwise comparison
    lr_scores = cv_results[0]['cv_summary']['f1_score_scores']
    rf_scores = cv_results[1]['cv_summary']['f1_score_scores']
    xgb_scores = cv_results[2]['cv_summary']['f1_score_scores']
    
    print("Cross-Validation F1 Scores:")
    print(f"LogisticRegression: {lr_scores}")
    print(f"RandomForest:       {rf_scores}")
    print(f"XGBoost:            {xgb_scores}")
    
    # Test significance between models
    print("\nPairwise Statistical Tests (Paired t-test):")
    print("-" * 45)
    
    # LogisticRegression vs RandomForest
    lr_vs_rf = evaluator.statistical_significance_test(lr_scores, rf_scores)
    print(f"LogisticRegression vs RandomForest:")
    print(f"  P-value: {lr_vs_rf['p_value']:.6f}")
    print(f"  Significant: {lr_vs_rf['significant']} (α=0.05)")
    print(f"  Mean difference: {lr_vs_rf['mean_diff']:.4f}")
    
    # RandomForest vs XGBoost
    rf_vs_xgb = evaluator.statistical_significance_test(rf_scores, xgb_scores)
    print(f"\nRandomForest vs XGBoost:")
    print(f"  P-value: {rf_vs_xgb['p_value']:.6f}")
    print(f"  Significant: {rf_vs_xgb['significant']} (α=0.05)")
    print(f"  Mean difference: {rf_vs_xgb['mean_diff']:.4f}")
    
    # LogisticRegression vs XGBoost
    lr_vs_xgb = evaluator.statistical_significance_test(lr_scores, xgb_scores)
    print(f"\nLogisticRegression vs XGBoost:")
    print(f"  P-value: {lr_vs_xgb['p_value']:.6f}")
    print(f"  Significant: {lr_vs_xgb['significant']} (α=0.05)")
    print(f"  Mean difference: {lr_vs_xgb['mean_diff']:.4f}")


def demonstrate_comprehensive_comparison():
    """Demonstrate comprehensive model comparison with significance."""
    print("\n" + "=" * 60)
    print("3. COMPREHENSIVE MODEL COMPARISON")
    print("=" * 60)
    
    evaluator = ModelEvaluator()
    cv_results = create_sample_cv_results()
    
    # Perform comprehensive comparison
    comparison_result = evaluator.compare_models_with_significance(
        cv_results, metric='f1_score'
    )
    
    print("Summary Statistics:")
    print("-" * 25)
    for stat in comparison_result['summary_statistics']:
        print(f"{stat['model_name']:18}: "
              f"F1 = {stat['f1_score_mean']:.3f} ± {stat['f1_score_std']:.3f}")
    
    print("\nPairwise Significance Matrix (p-values):")
    print("-" * 40)
    matrix = comparison_result['comparison_matrix']
    print(matrix.round(6))


def demonstrate_best_model_selection():
    """Demonstrate best model selection with different criteria."""
    print("\n" + "=" * 60)
    print("4. BEST MODEL SELECTION")
    print("=" * 60)
    
    evaluator = ModelEvaluator()
    model_results = create_sample_evaluation_results()
    
    # Default selection (balanced fraud detection metrics)
    print("Default Business Metrics Selection:")
    print("-" * 35)
    
    default_selection = evaluator.select_best_model(model_results)
    print(f"Best Model: {default_selection['best_model_name']}")
    print(f"Business Score: {default_selection['business_score']:.3f}")
    
    print("\nModel Ranking:")
    for rank_info in default_selection['model_ranking']:
        print(f"  {rank_info['rank']}. {rank_info['model_name']} "
              f"(score: {rank_info['business_score']:.3f})")
    
    # Precision-focused selection (minimize false positives)
    print("\n" + "-" * 40)
    print("Precision-Focused Selection (Minimize False Positives):")
    print("-" * 40)
    
    precision_metrics = {
        'precision': 0.6,   # High weight on precision
        'recall': 0.2,      # Lower weight on recall
        'f1_score': 0.1,    # Moderate weight on F1
        'auc_roc': 0.1      # Lower weight on AUC
    }
    
    precision_selection = evaluator.select_best_model(
        model_results, business_metrics=precision_metrics
    )
    print(f"Best Model: {precision_selection['best_model_name']}")
    print(f"Business Score: {precision_selection['business_score']:.3f}")
    
    # Recall-focused selection (catch more fraud)
    print("\n" + "-" * 40)
    print("Recall-Focused Selection (Catch More Fraud):")
    print("-" * 40)
    
    recall_metrics = {
        'precision': 0.2,   # Lower weight on precision
        'recall': 0.6,      # High weight on recall
        'f1_score': 0.1,    # Moderate weight on F1
        'auc_roc': 0.1      # Lower weight on AUC
    }
    
    recall_selection = evaluator.select_best_model(
        model_results, business_metrics=recall_metrics
    )
    print(f"Best Model: {recall_selection['best_model_name']}")
    print(f"Business Score: {recall_selection['business_score']:.3f}")
    
    # Selection with strict thresholds
    print("\n" + "-" * 40)
    print("Selection with Strict Quality Thresholds:")
    print("-" * 40)
    
    strict_thresholds = {
        'precision': 0.80,  # At least 80% precision
        'recall': 0.75,     # At least 75% recall
        'f1_score': 0.75    # At least 75% F1-score
    }
    
    strict_selection = evaluator.select_best_model(
        model_results, min_thresholds=strict_thresholds
    )
    
    if strict_selection['best_model']:
        print(f"Best Model: {strict_selection['best_model_name']}")
        print(f"Qualified Models: {strict_selection['qualified_models_count']}")
    else:
        print("No models met the strict quality thresholds!")
        print(f"Disqualified Models: {len(strict_selection['disqualified_models'])}")
        for disq in strict_selection['disqualified_models']:
            print(f"  - {disq['model_name']}: {', '.join(disq['failed_thresholds'])}")


def demonstrate_comprehensive_report():
    """Demonstrate comprehensive model selection report."""
    print("\n" + "=" * 60)
    print("5. COMPREHENSIVE SELECTION REPORT")
    print("=" * 60)
    
    evaluator = ModelEvaluator()
    model_results = create_sample_evaluation_results()
    cv_results = create_sample_cv_results()
    
    # Generate comprehensive report
    report = evaluator.generate_model_selection_report(
        model_results, cv_results
    )
    
    print(f"Report Generated: {report['timestamp']}")
    print(f"Models Evaluated: {report['models_evaluated']}")
    
    print(f"\nRecommended Model: {report['model_selection']['best_model_name']}")
    print(f"Business Score: {report['model_selection']['business_score']:.3f}")
    
    print(f"\nKey Insights:")
    for i, insight in enumerate(report['insights'], 1):
        print(f"  {i}. {insight}")
    
    # Show performance advantages
    advantages = report['model_selection']['performance_advantages']
    if advantages:
        print(f"\nPerformance Advantages of {report['model_selection']['best_model_name']}:")
        for metric, advantage in advantages.items():
            if advantage['improvement_percent'] > 1:  # Show improvements > 1%
                print(f"  - {metric.title()}: {advantage['improvement_percent']:.1f}% better "
                      f"({advantage['best_model_value']:.3f} vs {advantage['second_best_value']:.3f})")


def main():
    """Run the complete model comparison demonstration."""
    print("FRAUD DETECTION MODEL COMPARISON & SELECTION DEMO")
    print("=" * 60)
    print("This demo showcases the new model comparison and selection utilities")
    print("for fraud detection systems. It demonstrates:")
    print("• Performance comparison tables with rankings")
    print("• Statistical significance testing between models")
    print("• Business-metric-based model selection")
    print("• Comprehensive selection reports with insights")
    
    try:
        # Run all demonstrations
        demonstrate_comparison_table()
        demonstrate_statistical_significance()
        demonstrate_comprehensive_comparison()
        demonstrate_best_model_selection()
        demonstrate_comprehensive_report()
        
        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("The model comparison utilities provide comprehensive tools for:")
        print("• Comparing multiple fraud detection models")
        print("• Statistical validation of performance differences")
        print("• Business-driven model selection")
        print("• Automated reporting and insights")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())