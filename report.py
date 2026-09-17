def generate_report(result):
    print()
    print("=" * 45)
    print("       AI MODEL ROOT-CAUSE ANALYZER")
    print("=" * 45)

    print("\nDataset Summary")
    print("-" * 16)
    print(f"Rows              : {result['rows']}")
    print(f"Columns           : {result['columns']}")
    print(f"Missing Values    : {result['missing_values']}")
    print(f"Duplicate Rows    : {result['duplicate_rows']}")

    print("\nTarget Analysis")
    print("-" * 16)
    print(f"Target Column     : {result['target_column']}")
    print(f"Unique Classes    : {result['target_unique_values']}")

    print("\nClass Distribution")
    print("-" * 19)

    distribution = result.get("target_distribution", {})
    total = sum(distribution.values())

    for class_name, count in distribution.items():
        percentage = (count / total) * 100
        print(f"{class_name:<18}: {count} ({percentage:.1f}%)")

    print("\nDiagnosis")
    print("-" * 9)

    if result["status"] == "Warning":
        print("STATUS: WARNING")

        for cause in result["root_causes"]:
            print(f"[!] {cause}")

        if "imbalance_ratio" in result:
            print(f"\nImbalance Ratio   : {result['imbalance_ratio']}")

    else:
        print("STATUS: HEALTHY")
        print("No obvious data-related root causes detected.")

    print("\n" + "=" * 45)