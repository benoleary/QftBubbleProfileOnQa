from minimization.weight import WeightAccumulator


class TestBiasAccumulator():
    def test_linears_accumulate_correctly(self):
        test_accumulator = WeightAccumulator()
        # We use numbers which will have exact representations in binary so that
        # we can compare without needing a tolerance.
        test_accumulator.add_linears(
            {"a": 19.0, "b": 28.0, "c": 37.0, "d": 46.0}
        )
        test_accumulator.add_linears({"c": -12.0, "d": -48.0, "e": -13.0})

        expected_linears = {
            "a": 19.0, "b": 28.0, "c": 25.0, "d": -2.0, "e": -13.0
        }
        assert expected_linears == test_accumulator.linear_weights

    def test_quadratics_accumulate_correctly(self):
        test_accumulator = WeightAccumulator()
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
        assert expected_quadratics == test_accumulator.quadratic_weights

    def test_correct_form_for_QUBO(self):
        # TODO: implement
        pass
