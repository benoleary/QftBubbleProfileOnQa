from minimization.weight import BiasAccumulator


class TestBiasAccumulator():
    def test_linears_accumulate_correctly(self):
        test_accumulator = BiasAccumulator()
        # We use numbers which will have exact representations in binary so that
        # we can compare without needing a tolerance.
        test_accumulator.add_linears(
            {"a": 19.0, "b": 28.0, "c": 37.0, "d": 46.0}
        )
        test_accumulator.add_linears({"c": -12.0, "d": -48.0, "e": -13.0})

        expected_linears = {
            "a": 19.0, "b": 28.0, "c": 25.0, "d": -2.0, "e": -13.0
        }
        assert expected_linears == test_accumulator.linear_biases

    def test_quadratics_accumulate_correctly(self):
        test_accumulator = BiasAccumulator()
        # We use numbers which will have exact representations in binary so that
        # we can compare without needing a tolerance.
        test_accumulator.add_quadratics(
            {("a1", "a2"): 19.0, ("a1", "b2"): 28.0, ("c1", "c2"): 46.0}
        )
        test_accumulator.add_quadratics(
            {("a1", "a2"): 4.0, ("a2", "b2"): -5.0, ("c1", "c2"): 0.0}
        )

        expected_quadratics = {
            ("a1", "a2"): 23.0,
            ("a1", "b2"): 28.0,
            ("a2", "b2"): -5.0,
            ("c1", "c2"): 46.0
        }
        assert expected_quadratics == test_accumulator.quadratic_biases

    def test_copy_scales_and_is_deep_copy(self):
        source_accumulator = BiasAccumulator(
            initial_linears={"a": 19.0, "b": 28.0, "c": 37.0, "d": 46.0},
            initial_quadratics={
                ("a1", "a2"): 19.0, ("a1", "b2"): 28.0, ("c1", "c2"): 46.0
            }
        )
        scaled_accumulator = source_accumulator.create_scaled_copy(1.5)
        source_accumulator.add_linears({"d": -48.0, "e": -13.0})
        source_accumulator.add_quadratics({("a1", "a2"): 4.0})

        expected_source_linears = {
            "a": 19.0, "b": 28.0, "c": 37.0, "d": -2.0, "e": -13.0
        }
        expected_source_quadratics = {
                ("a1", "a2"): 23.0, ("a1", "b2"): 28.0, ("c1", "c2"): 46.0
        }
        expected_scaled_linears = {
            "a": 28.5, "b": 42.0, "c": 55.5, "d": 69.0
        }
        expected_scaled_quadratics = {
                ("a1", "a2"): 28.5, ("a1", "b2"): 42.0, ("c1", "c2"): 69.0
        }
        assert expected_source_linears == source_accumulator.linear_biases
        assert expected_source_quadratics == source_accumulator.quadratic_biases
        assert expected_scaled_linears == scaled_accumulator.linear_biases
        assert expected_scaled_quadratics == scaled_accumulator.quadratic_biases
