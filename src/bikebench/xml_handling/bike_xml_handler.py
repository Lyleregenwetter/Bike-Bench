from typing import Callable

from bs4 import BeautifulSoup

TEMPLATE_ENTRY = "<entry key='k'>1</entry>"


class BikeXmlHandler:
    """Stateful (AND NOT THREAD-SAFE) xml handler. Parses values using a supplied functional parser and
     alternatively converts an xml string into a string -> string dictionary"""
    XML_TAG = "entry"
    ATTRIBUTE = "key"
    PARENT_TAG = "properties"

    def __init__(self):
        self.xml_tree = None

    def get_all_entries_string(self):
        return self.get_all_entries().__str__()

    def get_entries_count(self):
        return len(self.get_all_entries())

    def set_xml(self, xml: str):
        self.xml_tree = self.generate_xml_tree(xml)

    def generate_xml_tree(self, xml: str):
        return BeautifulSoup(xml, "xml")

    def get_content_string(self):
        return self.xml_tree.__str__()

    def get_all_entries(self) -> dict:
        return self.xml_tree.find_all(self.XML_TAG)

    def copy_first_entry(self):
        fes = self.get_first_entry_string()
        return self.copy_entry(fes)

    def copy_entry(self, entry: str):
        new_tree_with_one_entry = self.generate_xml_tree(entry)
        entry_alone = self.strip_tree_of_needless_tags(new_tree_with_one_entry)
        return entry_alone

    def get_first_entry_string(self):
        return self.get_all_entries()[0].__str__()

    def strip_tree_of_needless_tags(self, new_tree):
        return new_tree.find_all(self.XML_TAG)[0]

    def add_new_entry(self, key: str, value: str):
        new_entry = self.copy_entry(TEMPLATE_ENTRY)
        new_entry[self.ATTRIBUTE] = key
        new_entry.find(string=new_entry.text).replace_with(value)
        self.xml_tree.find_all(self.PARENT_TAG)[0].append(new_entry)

    def find_entry_by_key(self, entry_key):
        for entry in self.get_all_entries():
            if entry[self.ATTRIBUTE] == entry_key:
                return entry

    def update_entry_key(self, entry, new_key):
        entry[self.ATTRIBUTE] = new_key

    def update_entry_value(self, entry, new_value):
        entry.find(string=entry.text).replace_with(new_value)

    def get_entries_dict(self):
        return {entry[self.ATTRIBUTE].strip(): entry.text for entry in self.get_all_entries()}

    def does_entry_exist(self, entry_key):
        if self.find_entry_by_key(entry_key):
            return True
        return False

    def remove_entry(self, entry):
        entry.decompose()

    def remove_all_entries(self):
        for entry in self.get_all_entries():
            self.remove_entry(entry)

    def set_entries_from_dict(self, entries_dict: dict):
        self.set_xml("")
        parent = self.xml_tree.new_tag("properties")
        self.xml_tree.append(parent)
        for key, value in entries_dict.items():
            self.add_new_entry(str(key), str(value))

    def update_entries_from_dict(self, entries_dict: dict):
        for key, value in entries_dict.items():
            self.add_or_update(key, value)

    def add_or_update(self, key, value):
        if self.key_exists(key):
            self.update_entry_value(self.find_entry_by_key(key), value)
        else:
            self.add_new_entry(key, value)

    def update_if_exists(self, key, value):
        if self.key_exists(key):
            self.update_entry_value(self.find_entry_by_key(key), value)

    def key_exists(self, key):
        return key in self.get_all_keys()

    def get_all_keys(self):
        return [entry[self.ATTRIBUTE] for entry in self.get_all_entries()]

    def get_parsed_entries(self, value_parser: Callable) -> dict:
        return {key: value_parser(value) for key, value in self.get_entries_dict().items()}
