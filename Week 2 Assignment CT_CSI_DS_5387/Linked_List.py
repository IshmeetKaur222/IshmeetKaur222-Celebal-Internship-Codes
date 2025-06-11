class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def add_node_to_end(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def print_list(self):
        if self.head is None:
            print("The list is currently empty.")
            return
        current = self.head
        elements = []
        while current:
            elements.append(str(current.data))
            current = current.next
        print(" -> ".join(elements))

    def delete_nth_node(self, n):
        if self.head is None:
            raise Exception("Deletion cannot be performed on an empty list.")

        if n == 1:
            self.head = self.head.next
            return

        current = self.head
        prev = None
        count = 1

        while current and count < n:
            prev = current
            current = current.next
            count += 1

        if current is None:
            raise IndexError(f"Error: Index {n} is out of bounds for the list.")

        prev.next = current.next

# Demonstration of functionality
if __name__ == "__main__":
    my_list = LinkedList()

    print("---")
    print("Initial list state:")
    my_list.print_list()

    print("\n---")
    print("Adding elements: 10, 20, 30, 40, 50")
    my_list.add_node_to_end(10)
    my_list.add_node_to_end(20)
    my_list.add_node_to_end(30)
    my_list.add_node_to_end(40)
    my_list.add_node_to_end(50)
    my_list.print_list()

    print("\n---")
    print("Attempting to delete the 3rd node (value 30):")
    try:
        my_list.delete_nth_node(3)
        my_list.print_list()
    except (Exception, IndexError) as e:
        print(f"An error occurred: {e}")

    print("\n---")
    print("Attempting to delete the 1st node (value 10):")
    try:
        my_list.delete_nth_node(1)
        my_list.print_list()
    except (Exception, IndexError) as e:
        print(f"An error occurred: {e}")

    print("\n---")
    print("Attempting to delete a node with an out-of-bounds index (index 10):")
    try:
        my_list.delete_nth_node(10)
        my_list.print_list()
    except (Exception, IndexError) as e:
        print(f"An error occurred: {e}")

    print("\n---")
    print("Attempting to delete the last remaining node (value 50):")
    try:
        my_list.delete_nth_node(3) # List is currently 20 -> 40 -> 50. The 3rd node is 50.
        my_list.print_list()
    except (Exception, IndexError) as e:
        print(f"An error occurred: {e}")

    print("\n---")
    print("Attempting to delete the final node (value 40):")
    try:
        my_list.delete_nth_node(1)
        my_list.print_list()
    except (Exception, IndexError) as e:
        print(f"An error occurred: {e}")

    print("\n---")
    print("Attempting to delete from an empty list:")
    empty_list = LinkedList()
    try:
        empty_list.delete_nth_node(1)
    except (Exception, IndexError) as e:
        print(f"An error occurred: {e}")

    print("\n---")
    print("Adding a single node and then deleting it:")
    single_node_list = LinkedList()
    single_node_list.add_node_to_end(100)
    single_node_list.print_list()
    try:
        single_node_list.delete_nth_node(1)
        single_node_list.print_list()
    except (Exception, IndexError) as e:
        print(f"An error occurred: {e}")