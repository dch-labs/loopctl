#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/pass.rs");
    t.compile_fail("tests/ui/missing_description.rs");
    t.compile_fail("tests/ui/unknown_attr.rs");
    t.compile_fail("tests/ui/bad_skip.rs");
    t.compile_fail("tests/ui/unsupported_type.rs");
    t.compile_fail("tests/ui/enum_input.rs");
    t.compile_fail("tests/ui/generic_struct.rs");
    t.compile_fail("tests/ui/cow_bytes.rs");
    t.compile_fail("tests/ui/name_disagreement.rs");
    t.compile_fail("tests/ui/default_without_serde.rs");
    t.compile_fail("tests/ui/bad_handler.rs");
}
